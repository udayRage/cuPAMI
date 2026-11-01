import argparse
import csv
import itertools
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def read_database(path, fmt='fuzzy', repeat=1, duplicate_policy='max'):

    rows = defaultdict(dict)
    count = 0
    def add(t, item, region, value):
        v = float(value)
        if not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f'Invalid membership {value!r}')
        key = (str(item), str(region))
        if t in rows[key]:
            if rows[key][t] == v:
                return
            if duplicate_policy == 'error':
                raise ValueError(f'Conflicting duplicate term {key} in transaction {t}')
            v = max(rows[key][t], v)
        rows[key][t] = v
    with open(path, newline='') as f:
        if fmt == 'triples':
            tids = {}
            for row in csv.DictReader(f):
                t = tids.setdefault(row['transaction'], len(tids))
                add(t, row['item'], row['region'], row['membership'])
            count = len(tids)
        else:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                fields = line.strip().split(':')
                if len(fields) != 2:
                    raise ValueError(f'Line {line_number}: expected terms : memberships; use --format triples for CSV triples')
                terms = fields[0].replace(',', ' ').split()
                values = fields[1].replace(',', ' ').split()
                if len(terms) != len(values):
                    raise ValueError(f'Line {line_number}: different term and membership counts')
                for term, value in zip(terms, values):
                    if '.' not in term:
                        raise ValueError(f'Line {line_number}: expected item.region, got {term!r}')
                    item, region = term.rsplit('.', 1)
                    add(count, item, region, value)
                count += 1
    result = {}
    for term, values in rows.items():
        pairs = sorted((t, v) for t, v in values.items() if v > 0)
        tids = np.array([t for t, _ in pairs], dtype=np.int64)
        mem = np.array([v for _, v in pairs], dtype=np.float64)
        result[term] = (np.concatenate([tids + r * count for r in range(repeat)]), np.tile(mem, repeat))
    return result, count * repeat


def align(left, right, xp):

    if not len(left) or not len(right):
        return xp.empty(0, dtype=xp.int64), xp.empty(0, dtype=xp.int64)
    pos = xp.searchsorted(right, left)
    safe = xp.minimum(pos, len(right) - 1)
    keep = (pos < len(right)) & (right[safe] == left)
    return xp.nonzero(keep)[0], pos[keep]


def scalar(value):
    return float(value.item())


CUDA_SOURCE = r"""
#define MAX_PAIR_COUNTERS 32
#define MAX_EXTENSION_COUNTERS 32
typedef int support_counter_t;
#define CSR_PACKED 0
#define SPARSE_PIVOT 0
__device__ __forceinline__ long long grouped_lower_bound(const int* rt,long long nr,int tid,const int* samples,long long step,bool indexed){
    long long lo=0,hi=nr;
    if(indexed){
        int a=0,b=128;
        while(a<b){int m=(a+b)/2;if(samples[m]<tid)a=m+1;else b=m;}
        if(a<128 && (long long)a*step<nr && samples[a]==tid)return (long long)a*step;
        lo=a?(long long)(a-1)*step+1:0;
        hi=a<128 && (long long)a*step<nr?(long long)a*step:nr;
    }
    while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}
    return lo;
}
extern "C" __global__ void singletons(
    const short* values, const long long* offsets, long long* sums) {
    int job=blockIdx.x, lane=threadIdx.x;
    __shared__ long long partial[128];
    long long total=0;
    for(long long k=offsets[job]+lane;k<offsets[job+1];k+=128) total+=values[k];
    partial[lane]=total; __syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(lane<stride) partial[lane]+=partial[lane+stride];
        __syncthreads();
    }
    if(!lane) sums[job]=partial[0];
}
extern "C" __global__ void support_jobs(
    const unsigned long long* pointers,const long long* sizes,const long long* offsets,const int* columns,long long* sums){
    int job=blockIdx.x,lane=threadIdx.x;
    const int* lt=(const int*)pointers[4*job];const short* lv=(const short*)pointers[4*job+1];
    const int* rt=(const int*)pointers[4*job+2];const short* rv=(const short*)pointers[4*job+3];
    long long nl=sizes[4*job],nr=sizes[4*job+1];int lw=sizes[4*job+2],rw=sizes[4*job+3];
    int first=offsets[job],nc=offsets[job+1]-first;bool swapped=nl>nr;
    if(swapped){const int* t=lt;lt=rt;rt=t;const short* v=lv;lv=rv;rv=v;long long n=nl;nl=nr;nr=n;int w=lw;lw=rw;rw=w;}
    __shared__ int lc[32],rc[32],samples[128];
    if(lane<nc){lc[lane]=columns[2*(first+lane)+(swapped?1:0)];rc[lane]=columns[2*(first+lane)+(swapped?0:1)];}__syncthreads();
    bool indexed=SPARSE_PIVOT && nr>=512 && nl>=32;long long step=(nr+127)/128;
    if(indexed){for(int p=lane;p<128;p+=blockDim.x){long long at=p*step;samples[p]=at<nr?rt[at]:2147483647;}__syncthreads();}
    support_counter_t local[MAX_EXTENSION_COUNTERS];
#pragma unroll
    for(int c=0;c<MAX_EXTENSION_COUNTERS;c++)local[c]=0;
    for(long long k=lane;k<nl;k+=blockDim.x){
        int tid=lt[k];long long pos=grouped_lower_bound(rt,nr,tid,samples,step,indexed);
        if(pos<nr && rt[pos]==tid){
#pragma unroll
            for(int c=0;c<MAX_EXTENSION_COUNTERS;c++)if(c<nc){int a=lv[k*lw+lc[c]],b=rv[pos*rw+rc[c]];local[c]+=a<b?a:b;}
        }
    }
    extern __shared__ support_counter_t partial[];
    int warp=lane/32,wlane=lane%32,warps=blockDim.x/32;
#pragma unroll
    for(int c=0;c<MAX_EXTENSION_COUNTERS;c++)if(c<nc){
        support_counter_t value=local[c];
        for(int s=16;s;s/=2)value+=__shfl_down_sync(0xffffffff,value,s);
        if(!wlane)partial[c*warps+warp]=value;
    }
    __syncthreads();
    if(lane<nc){support_counter_t value=0;for(int w=0;w<warps;w++)value+=partial[lane*warps+w];sums[first+lane]=value;}
}
extern "C" __global__ void pair_support(
    const unsigned long long* baseptrs,const long long* baselengths,
    const unsigned long long* termptrs,const long long* termlengths,
    const int* widths,int n,int maxwidth,long long firstpair,int grouped,long long* sums){
    int lane=threadIdx.x,slots=maxwidth*maxwidth;
    long long localpair=grouped?blockIdx.x:blockIdx.x/slots;int slot=grouped?0:blockIdx.x%slots;
    long long p=firstpair+localpair;
    __shared__ long long pi,pj;
    if(!lane){long long i=(long long)(((2.0*n-1)-sqrt((2.0*n-1)*(2.0*n-1)-8.0*p))/2.0);
        while(i*(2LL*n-i-1)/2>p)i--;while((i+1)*(2LL*n-i-2)/2<=p)i++;
        pi=i;pj=i+1+p-i*(2LL*n-i-1)/2;}
    __syncthreads();long long i=pi,j=pj;
    int wi=widths[i],wj=widths[j],ca=slot/maxwidth,cb=slot%maxwidth;
    if(!grouped && (ca>=wi || cb>=wj))return;
    const int *lt,*rt;const short *lv,*rv;long long nl,nr;int lw,rw,nc;
    if(grouped){lt=(const int*)baseptrs[2*i];lv=(const short*)baseptrs[2*i+1];rt=(const int*)baseptrs[2*j];rv=(const short*)baseptrs[2*j+1];nl=baselengths[i];nr=baselengths[j];lw=wi;rw=wj;nc=wi*wj;}
    else{long long a=i*maxwidth+ca,b=j*maxwidth+cb;lt=(const int*)termptrs[2*a];lv=(const short*)termptrs[2*a+1];rt=(const int*)termptrs[2*b];rv=(const short*)termptrs[2*b+1];nl=termlengths[a];nr=termlengths[b];lw=rw=nc=1;}
    bool swapped=nl>nr;
    if(swapped){const int* t=lt;lt=rt;rt=t;const short* v=lv;lv=rv;rv=v;long long x=nl;nl=nr;nr=x;int w=lw;lw=rw;rw=w;}
    __shared__ int lc[32],rc[32],samples[128];
    if(lane<nc){int a=grouped?lane/wj:0,b=grouped?lane%wj:0;lc[lane]=swapped?b:a;rc[lane]=swapped?a:b;}__syncthreads();
    bool indexed=SPARSE_PIVOT && nr>=512 && nl>=32;long long step=(nr+127)/128;
    if(indexed){for(int q=lane;q<128;q+=blockDim.x){long long at=q*step;samples[q]=at<nr?rt[at]:2147483647;}__syncthreads();}
    support_counter_t local[MAX_PAIR_COUNTERS];
#pragma unroll
    for(int c=0;c<MAX_PAIR_COUNTERS;c++)local[c]=0;
    for(long long k=lane;k<nl;k+=blockDim.x){
        int tid=lt[k];long long pos=grouped_lower_bound(rt,nr,tid,samples,step,indexed);
        if(pos<nr && rt[pos]==tid){
#pragma unroll
            for(int c=0;c<MAX_PAIR_COUNTERS;c++)if(c<nc){int a=lv[k*lw+lc[c]],b=rv[pos*rw+rc[c]];local[c]+=a<b?a:b;}
        }
    }
    extern __shared__ support_counter_t partial[];int warp=lane/32,wlane=lane%32,warps=blockDim.x/32;
#pragma unroll
    for(int c=0;c<MAX_PAIR_COUNTERS;c++)if(c<nc){support_counter_t value=local[c];for(int s=16;s;s/=2)value+=__shfl_down_sync(0xffffffff,value,s);if(!wlane)partial[c*warps+warp]=value;}
    __syncthreads();
    if(lane<nc){support_counter_t value=0;for(int w=0;w<warps;w++)value+=partial[lane*warps+w];int out=grouped?(lane/wj)*maxwidth+lane%wj:slot;sums[localpair*slots+out]=value;}
}
extern "C" __global__ void materialize(
    const unsigned long long* pointers, const long long* sizes,
    const int* columns, const long long* offsets,
    int* tids, short* values) {
    int job=blockIdx.x;
    const int* lt=(const int*)pointers[4*job];
    const short* lv=(const short*)pointers[4*job+1];
    const int* rt=(const int*)pointers[4*job+2];
    const short* rv=(const short*)pointers[4*job+3];
    long long nl=sizes[4*job], nr=sizes[4*job+1];
    int lw=sizes[4*job+2], rw=sizes[4*job+3];
    int lc=columns[2*job], rc=columns[2*job+1];
    for(long long k=threadIdx.x;k<nl;k+=blockDim.x) {
        long long tid=lt[k], lo=0, hi=nr;
        while(lo<hi) {long long m=(lo+hi)/2; if(rt[m]<tid)lo=m+1;else hi=m;}
        long long v=0;
        if(lo<nr && rt[lo]==tid) {
            long long a=lv[k*lw+lc];
            long long b=rv[lo*rw+rc];
            v=a<b?a:b;
        }
        tids[offsets[job]+k]=tid;
        values[offsets[job]+k]=v;
    }
}
extern "C" __global__ void grouped_flags(
    const short* values, const long long* metadata, const int* columns,
    int colstride, unsigned char* flags) {
    int job=blockIdx.x;
    long long voff=metadata[6*job+1], width=metadata[6*job+2];
    long long n=metadata[6*job+3], nr=metadata[6*job+4], dense=metadata[6*job+5];
    for(long long k=threadIdx.x;k<n;k+=blockDim.x) {
        bool keep=false;
        for(int r=0;r<nr;r++) keep|=values[voff+k*width+columns[job*colstride+r]]>0;
        flags[dense+k]=keep;
    }
}
extern "C" __global__ void grouped_copy(
    const int* tids, const short* values, const long long* metadata,
    const int* columns, int colstride, const unsigned char* flags,
    const int* scan, const long long* output_offsets, int* output_tids, short* output_values) {
    int job=blockIdx.x;
    long long toff=metadata[6*job], voff=metadata[6*job+1], width=metadata[6*job+2];
    long long n=metadata[6*job+3], nr=metadata[6*job+4], dense=metadata[6*job+5];
    long long outrow=output_offsets[2*job], outval=output_offsets[2*job+1];
    for(long long k=threadIdx.x;k<n;k+=blockDim.x) if(flags[dense+k]) {
        long long row=scan[dense+k+1]-1;
        output_tids[row]=tids[toff+k];
        for(int r=0;r<nr;r++) output_values[outval+(row-outrow)*nr+r]=
            values[voff+k*width+columns[job*colstride+r]];
    }
}

extern "C" __global__ void extension_counts(
    const int* patterns, int length, long long rows, int nterms,
    const unsigned char* pairs, const long long* noff, const int* neighbors,
    const int* bases, int grouped, long long* nc, long long* nj) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
    int chosen=patterns[row*length];
    for(int k=1;k<length;k++) {
        int t=patterns[row*length+k];
        if(noff[t+1]-noff[t]<noff[chosen+1]-noff[chosen])chosen=t;
    }
    int last=bases[patterns[row*length+length-1]], previous=-1;
    long long candidates=0,jobs=0;
    for(long long p=noff[chosen];p<noff[chosen+1];p++) {
        int t=neighbors[p],base=bases[t];
        if(base<=last)continue;
        bool valid=true;
        for(int k=0;k<length;k++) if(!pairs[(long long)patterns[row*length+k]*nterms+t]){valid=false;break;}
        if(!valid)continue;
        candidates++;
        if(!grouped || base!=previous){jobs++;previous=base;}
    }
    nc[row]=candidates;nj[row]=jobs;
}
extern "C" __global__ void extension_fill(
    const int* patterns, int length, long long rows, int nterms,
    const unsigned char* pairs, const long long* noff, const int* neighbors,
    const int* bases, int grouped, const long long* coff, const long long* joff,
    int* terms, long long* parents, int* rights, long long* offsets) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
    int chosen=patterns[row*length];
    for(int k=1;k<length;k++) {
        int t=patterns[row*length+k];
        if(noff[t+1]-noff[t]<noff[chosen+1]-noff[chosen])chosen=t;
    }
    int last=bases[patterns[row*length+length-1]], previous=-1;
    long long candidate=coff[row],job=joff[row];
    for(long long p=noff[chosen];p<noff[chosen+1];p++) {
        int t=neighbors[p],base=bases[t];
        if(base<=last)continue;
        bool valid=true;
        for(int k=0;k<length;k++)if(!pairs[(long long)patterns[row*length+k]*nterms+t]){valid=false;break;}
        if(!valid)continue;
        if(!grouped || base!=previous){parents[job]=row;rights[job]=grouped?base:t;offsets[job]=candidate;job++;previous=base;}
        terms[candidate++]=t;
    }
}
extern "C" __global__ void extension_descriptors(
    long long jobs, int grouped, const int* terms, const long long* parents,
    const int* rights, const long long* offsets, const long long* current_offsets,
    unsigned long long tids, unsigned long long values,
    const unsigned long long* rptr, const long long* rlength, const long long* rwidth,
    const unsigned long long* sptr, const long long* slength, const int* region,
    unsigned long long* pointers, long long* sizes, int* columns) {
    long long job=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(job>=jobs)return;
    long long parent=parents[job],first=offsets[job],end=offsets[job+1];
    int right=rights[job];
    bool single=grouped && end-first==1;
    pointers[4*job]=tids+4ULL*current_offsets[parent];
    pointers[4*job+1]=values+2ULL*current_offsets[parent];
    pointers[4*job+2]=single?sptr[2*terms[first]]:rptr[2*right];
    pointers[4*job+3]=single?sptr[2*terms[first]+1]:rptr[2*right+1];
    sizes[4*job]=current_offsets[parent+1]-current_offsets[parent];
    sizes[4*job+1]=single?slength[terms[first]]:rlength[right];
    sizes[4*job+2]=1;sizes[4*job+3]=single?1:rwidth[right];
    for(long long c=first;c<end;c++){columns[2*c]=0;columns[2*c+1]=grouped && !single?region[terms[c]]:0;}
}

extern "C" __global__ void flat_support_jobs(
    const unsigned long long* pointers, const long long* sizes,
    const long long* offsets, const int* columns, long long* sums) {
    int job=blockIdx.x,lane=threadIdx.x;
    const int* lt=(const int*)pointers[4*job];const short* lv=(const short*)pointers[4*job+1];
    const int* rt=(const int*)pointers[4*job+2];const short* rv=(const short*)pointers[4*job+3];
    long long nl=sizes[4*job],nr=sizes[4*job+1],out=offsets[job];
    if(!lane)sums[out]=0;__syncthreads();
    for(long long k=lane;k<nl;k+=blockDim.x){
        int tid=lt[k];long long lo=0,hi=nr;
        while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}
        if(lo<nr && rt[lo]==tid){int a=lv[k],b=rv[lo],v=a<b?a:b;if(v)atomicAdd((unsigned long long*)&sums[out],(unsigned long long)v);}
    }
}
extern "C" __global__ void flat_pair_support(
    const unsigned long long* pointers, const long long* lengths, const int* bases,
    int n,long long first,long long* sums) {
    int lane=threadIdx.x;
    long long p=first+blockIdx.x;
    __shared__ long long si,sj;
    if(!lane){
        long long i=(long long)(((2.0*n-1)-sqrt((2.0*n-1)*(2.0*n-1)-8.0*p))/2.0);
        while(i*(2LL*n-i-1)/2>p)i--;
        while((i+1)*(2LL*n-i-2)/2<=p)i++;
        si=i;sj=i+1+p-i*(2LL*n-i-1)/2;
    }
    __syncthreads();long long i=si,j=sj;
    if(bases[i]==bases[j])return;
    const int* lt=(const int*)pointers[2*i];const short* lv=(const short*)pointers[2*i+1];
    const int* rt=(const int*)pointers[2*j];const short* rv=(const short*)pointers[2*j+1];
    long long nl=lengths[i],nr=lengths[j];
    for(long long k=lane;k<nl;k+=blockDim.x){
        int tid=lt[k];long long lo=0,hi=nr;
        while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}
        if(lo<nr && rt[lo]==tid){int a=lv[k],b=rv[lo],v=a<b?a:b;if(v)atomicAdd((unsigned long long*)&sums[blockIdx.x],(unsigned long long)v);}
    }
}

__device__ bool csv_sep(unsigned char c){return c==' ' || c=='\t' || c=='\r' || c==',';}
extern "C" __global__ void csv_counts(const unsigned char* buf,const long long* start,const long long* colon,long long n,long long* counts){
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    long long count=0;for(long long p=start[row];p<colon[row];p++)if(buf[p]=='.')count++;
    counts[row]=count;
}
extern "C" __global__ void csv_parse(const unsigned char* buf,const long long* start,const long long* colon,const long long* end,
    const long long* offsets,long long n,int scale,int* tids,int* bases,int* labels,short* values,int* error){
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    long long first=offsets[row],count=offsets[row+1]-first,p=start[row];
    for(long long k=0;k<count;k++){
        while(p<colon[row] && csv_sep(buf[p]))p++;
        long long item=0;bool digit=false;
        while(p<colon[row] && buf[p]>='0' && buf[p]<='9'){item=item*10+buf[p++]-'0';digit=true;}
        if(!digit || p>=colon[row] || buf[p++]!='.' || item>2147483647LL){atomicExch(error,1);return;}
        int lab=0,len=0;
        while(p<colon[row] && buf[p]>='A' && buf[p]<='Z'){lab=(lab<<5)|(buf[p++]-'A'+1);len++;if(len>6){atomicExch(error,1);return;}}
        if(!len || (p<colon[row] && !csv_sep(buf[p]))){atomicExch(error,1);return;}
        tids[first+k]=row;bases[first+k]=item;labels[first+k]=lab;
    }
    while(p<colon[row] && csv_sep(buf[p]))p++;
    if(p!=colon[row]){atomicExch(error,1);return;}
    p=colon[row]+1;
    for(long long k=0;k<count;k++){
        while(p<end[row] && csv_sep(buf[p]))p++;
        double value=0,divisor=1;bool digit=false,dot=false;
        while(p<end[row] && ((buf[p]>='0' && buf[p]<='9') || buf[p]=='.')){
            if(buf[p]=='.'){if(dot){atomicExch(error,2);return;}dot=true;p++;continue;}
            digit=true;value=value*10+buf[p++]-'0';if(dot)divisor*=10;
        }
        value/=divisor;double scaled=value*scale;long long tick=(long long)llrint(scaled);
        if(!digit || value<0 || value>1 || fabs(scaled-tick)>1e-7 || (p<end[row] && !csv_sep(buf[p]))){atomicExch(error,2);return;}
        values[first+k]=tick;
    }
    while(p<end[row] && csv_sep(buf[p]))p++;
    if(p!=end[row])atomicExch(error,2);
}
extern "C" __global__ void dedup_memberships(const short* values,const long long* starts,long long groups,long long records,
    int strict,short* output,int* error){
    long long group=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(group>=groups)return;
    long long first=starts[group],end=group+1<groups?starts[group+1]:records;
    int maximum=values[first];
    for(long long p=first+1;p<end;p++){if(strict && values[p]!=values[first])atomicExch(error,1);maximum=maximum>values[p]?maximum:values[p];}
    output[group]=maximum;
}
extern "C" __global__ void repeat_occurrences(const int* tids,const short* values,const long long* offsets,const long long* value_offsets,
    const long long* widths,int repeat,int transactions,const long long* output_offsets,const long long* output_value_offsets,
    int* output_tids,short* output_values){
    int job=blockIdx.x;long long n=offsets[job+1]-offsets[job],width=widths[job];
    for(long long k=threadIdx.x;k<n*repeat;k+=blockDim.x){
        long long original=k%n,copy=k/n;
        output_tids[output_offsets[job]+k]=tids[offsets[job]+original]+copy*transactions;
        for(int c=0;c<width;c++)output_values[output_value_offsets[job]+k*width+c]=values[value_offsets[job]+original*width+c];
    }
}

extern "C" __global__ void byte_positions(const unsigned char* buf,const int* prefix,long long bytes,int target,long long* output){
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p<bytes && buf[p]==target)output[prefix[p]-1]=p;
}
extern "C" __global__ void record_supports(const long long* codes,const short* values,long long records,int wide,void* output){
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(row<records){if(wide)atomicAdd((unsigned long long*)output+codes[row],(unsigned long long)values[row]);
                  else atomicAdd((int*)output+codes[row],(int)values[row]);}
}

__device__ long long serial_intersection(const int* lt,const short* lv,long long nl,const int* rt,const short* rv,long long nr){
    if(!nl || !nr)return 0;
    long long i=0,j=0,total=0;int a=lt[0],b=rt[0];
    while(i<nl && j<nr){
        if(a==b){int u=lv[i],v=rv[j];total+=u<v?u:v;i++;j++;if(i<nl)a=lt[i];if(j<nr)b=rt[j];}
        else if(a<b){i++;if(i<nl)a=lt[i];}
        else{j++;if(j<nr)b=rt[j];}
    }
    return total;
}
extern "C" __global__ void flat_support_serial(const unsigned long long* ptrs,const long long* sizes,
    const long long* offsets,long long* sums,long long jobs){
    long long job=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(job>=jobs)return;
    sums[offsets[job]]=serial_intersection((const int*)ptrs[4*job],(const short*)ptrs[4*job+1],sizes[4*job],
                                         (const int*)ptrs[4*job+2],(const short*)ptrs[4*job+3],sizes[4*job+1]);
}
extern "C" __global__ void flat_pair_serial(const unsigned long long* ptrs,const long long* lengths,const int* bases,
    int n,long long first,long long jobs,long long* sums){
    long long job=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(job>=jobs)return;
    long long p=first+job;
    long long i=(long long)(((2.0*n-1)-sqrt((2.0*n-1)*(2.0*n-1)-8.0*p))/2.0);
    while(i*(2LL*n-i-1)/2>p)i--;
    while((i+1)*(2LL*n-i-2)/2<=p)i++;
    long long j=i+1+p-i*(2LL*n-i-1)/2;
    if(bases[i]==bases[j])return;
    sums[job]=serial_intersection((const int*)ptrs[2*i],(const short*)ptrs[2*i+1],lengths[i],
                                (const int*)ptrs[2*j],(const short*)ptrs[2*j+1],lengths[j]);
}

extern "C" __global__ void csv_position_counts(const unsigned char* buf,long long bytes,int* counts){
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int a=__ballot_sync(0xffffffff,p<bytes && buf[p]==10);
    unsigned int b=__ballot_sync(0xffffffff,p<bytes && buf[p]==58);
    __shared__ int ca[8],cb[8];
    if(!lane){ca[warp]=__popc(a);cb[warp]=__popc(b);}__syncthreads();
    if(!threadIdx.x){int x=0,y=0;for(int w=0;w<8;w++){x+=ca[w];y+=cb[w];}counts[2*blockIdx.x]=x;counts[2*blockIdx.x+1]=y;}
}
extern "C" __global__ void csv_position_write(const unsigned char* buf,long long bytes,const int* aoff,const int* boff,
    long long* ends,long long* colons){
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    bool isa=p<bytes && buf[p]==10,isb=p<bytes && buf[p]==58;
    unsigned int a=__ballot_sync(0xffffffff,isa),b=__ballot_sync(0xffffffff,isb);
    __shared__ int ca[8],cb[8];
    if(!lane){ca[warp]=__popc(a);cb[warp]=__popc(b);}__syncthreads();
    int xa=blockIdx.x?aoff[blockIdx.x-1]:0,xb=blockIdx.x?boff[blockIdx.x-1]:0;
    for(int w=0;w<warp;w++){xa+=ca[w];xb+=cb[w];}
    unsigned int before=(1u<<lane)-1u;
    if(isa)ends[xa+__popc(a&before)]=p;
    if(isb)colons[xb+__popc(b&before)]=p;
}

__device__ void csr_pair_ids(long long p,int n,long long& i,long long& j){
    i=(long long)(((2.0*n-1)-sqrt((2.0*n-1)*(2.0*n-1)-8.0*p))/2.0);
    while(i*(2LL*n-i-1)/2>p)i--;
    while((i+1)*(2LL*n-i-2)/2<=p)i++;
    j=i+1+p-i*(2LL*n-i-1)/2;
}
__device__ __forceinline__ void csr_store(int* tids,short* values,long long at,int tid,int value){
#if CSR_PACKED
    tids[at]=(int)(((unsigned int)tid<<4)|(unsigned int)value);
#else
    tids[at]=tid;values[at]=value;
#endif
}
__device__ long long csr_serial_merge(const int* lt,const short* lv,long long nl,const int* rt,const short* rv,long long nr,
    int* output_tids,short* output_values,long long output){
    long long i=0,j=0,count=0;
    while(i<nl && j<nr){
        int a=lt[i],b=rt[j];
        if(a==b){if(output_tids){int u=lv[i],v=rv[j];csr_store(output_tids,output_values,output+count,a,u<v?u:v);}count++;i++;j++;}
        else if(a<b)i++;else j++;
    }
    return count;
}
__device__ long long csr_parallel_count(const int* lt,long long nl,const int* rt,long long nr){
    long long local=0;
    for(long long k=threadIdx.x;k<nl;k+=blockDim.x){
        int tid=lt[k];long long lo=0,hi=nr;
        while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}
        local+=lo<nr && rt[lo]==tid;
    }
    __shared__ long long scratch[256];scratch[threadIdx.x]=local;__syncthreads();
    for(int s=blockDim.x/2;s;s/=2){if(threadIdx.x<s)scratch[threadIdx.x]+=scratch[threadIdx.x+s];__syncthreads();}
    return scratch[0];
}
__device__ void csr_parallel_write(const int* lt,const short* lv,long long nl,const int* rt,const short* rv,long long nr,
    int* output_tids,short* output_values,long long output){
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    __shared__ int warp_counts[8],warp_offsets[8];
    __shared__ long long written,batch;
    if(!threadIdx.x)written=0;__syncthreads();
    for(long long tile=0;tile<nl;tile+=blockDim.x){
        long long k=tile+threadIdx.x;bool found=false;int tid=0,value=0;
        if(k<nl){tid=lt[k];long long lo=0,hi=nr;
            while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}
            found=lo<nr && rt[lo]==tid;if(found){int a=lv[k],b=rv[lo];value=a<b?a:b;}}
        unsigned int mask=__ballot_sync(0xffffffff,found);
        if(!lane)warp_counts[warp]=__popc(mask);__syncthreads();
        if(!threadIdx.x){int n=0;for(int w=0;w<blockDim.x/32;w++){warp_offsets[w]=n;n+=warp_counts[w];}batch=written;written+=n;}
        __syncthreads();
        if(found){long long at=output+batch+warp_offsets[warp]+__popc(mask&((1u<<lane)-1u));csr_store(output_tids,output_values,at,tid,value);}
    }
}
extern "C" __global__ void csr_pair_count(const unsigned long long* ptrs,const long long* lengths,const int* bases,
    int n,long long first,long long jobs,int parallel,long long* counts){
    long long job=parallel?blockIdx.x:(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(job>=jobs)return;long long i,j;
    __shared__ long long pi,pj;
    if(parallel){if(!threadIdx.x)csr_pair_ids(first+job,n,pi,pj);__syncthreads();i=pi;j=pj;}
    else csr_pair_ids(first+job,n,i,j);
    if(bases[i]==bases[j]){if(!parallel || !threadIdx.x)counts[job]=0;return;}
    const int* lt=(const int*)ptrs[2*i];const int* rt=(const int*)ptrs[2*j];
    long long result=parallel?csr_parallel_count(lt,lengths[i],rt,lengths[j]):
        csr_serial_merge(lt,0,lengths[i],rt,0,lengths[j],0,0,0);
    if(!parallel || !threadIdx.x)counts[job]=result;
}
extern "C" __global__ void csr_pair_write(const unsigned long long* ptrs,const long long* lengths,const int* bases,
    int n,long long first,long long jobs,int parallel,const long long* offsets,int* output_tids,short* output_values){
    long long job=parallel?blockIdx.x:(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(job>=jobs)return;long long i,j;
    __shared__ long long pi,pj;
    if(parallel){if(!threadIdx.x)csr_pair_ids(first+job,n,pi,pj);__syncthreads();i=pi;j=pj;}
    else csr_pair_ids(first+job,n,i,j);
    if(bases[i]==bases[j])return;
    const int* lt=(const int*)ptrs[2*i];const short* lv=(const short*)ptrs[2*i+1];
    const int* rt=(const int*)ptrs[2*j];const short* rv=(const short*)ptrs[2*j+1];
    if(parallel)csr_parallel_write(lt,lv,lengths[i],rt,rv,lengths[j],output_tids,output_values,offsets[job]);
    else csr_serial_merge(lt,lv,lengths[i],rt,rv,lengths[j],output_tids,output_values,offsets[job]);
}
extern "C" __global__ void csr_job_count(const unsigned long long* ptrs,const long long* sizes,long long jobs,int parallel,long long* counts){
    long long job=parallel?blockIdx.x:(long long)blockIdx.x*blockDim.x+threadIdx.x;if(job>=jobs)return;
    const int* lt=(const int*)ptrs[4*job];const int* rt=(const int*)ptrs[4*job+2];
    long long result=parallel?csr_parallel_count(lt,sizes[4*job],rt,sizes[4*job+1]):csr_serial_merge(lt,0,sizes[4*job],rt,0,sizes[4*job+1],0,0,0);
    if(!parallel || !threadIdx.x)counts[job]=result;
}
extern "C" __global__ void csr_job_write(const unsigned long long* ptrs,const long long* sizes,long long jobs,int parallel,
    const long long* offsets,int* output_tids,short* output_values){
    long long job=parallel?blockIdx.x:(long long)blockIdx.x*blockDim.x+threadIdx.x;if(job>=jobs)return;
    const int* lt=(const int*)ptrs[4*job];const short* lv=(const short*)ptrs[4*job+1];
    const int* rt=(const int*)ptrs[4*job+2];const short* rv=(const short*)ptrs[4*job+3];
    if(parallel)csr_parallel_write(lt,lv,sizes[4*job],rt,rv,sizes[4*job+1],output_tids,output_values,offsets[job]);
    else csr_serial_merge(lt,lv,sizes[4*job],rt,rv,sizes[4*job+1],output_tids,output_values,offsets[job]);
}
extern "C" __global__ void csr_supports(const int* tids,const short* values,const long long* offsets,long long jobs,long long* supports){
    long long job=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(job>=jobs)return;
    long long support=0;for(long long k=offsets[job];k<offsets[job+1];k++){
#if CSR_PACKED
        support+=(unsigned int)tids[k]&15u;
#else
        support+=values[k];
#endif
    }supports[job]=support;
}
extern "C" __global__ void csr_select(const int* tids,const short* values,const long long* offsets,const long long* selected,
    const long long* output_offsets,int* output_tids,short* output_values){
    long long job=blockIdx.x,candidate=selected[job],n=offsets[candidate+1]-offsets[candidate];
    for(long long k=threadIdx.x;k<n;k+=blockDim.x){
#if CSR_PACKED
        unsigned int word=(unsigned int)tids[offsets[candidate]+k];
        output_tids[output_offsets[job]+k]=word>>4;output_values[output_offsets[job]+k]=word&15u;
#else
        output_tids[output_offsets[job]+k]=tids[offsets[candidate]+k];output_values[output_offsets[job]+k]=values[offsets[candidate]+k];
#endif
    }
}

__device__ long long csr_warp_count(const int* lt,long long nl,const int* rt,long long nr){
    if(nl>nr){const int* tmp=lt;lt=rt;rt=tmp;long long n=nl;nl=nr;nr=n;}
    int lane=threadIdx.x&31,count=0;
    for(long long k=lane;k<nl;k+=32){int tid=lt[k];long long lo=0,hi=nr;
        while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}count+=lo<nr && rt[lo]==tid;}
    for(int s=16;s;s/=2)count+=__shfl_down_sync(0xffffffff,count,s);
    return count;
}
__device__ void csr_warp_write(const int* lt,const short* lv,long long nl,const int* rt,const short* rv,long long nr,
    int* output_tids,short* output_values,long long output){
    if(nl>nr){const int* t=lt;lt=rt;rt=t;const short* v=lv;lv=rv;rv=v;long long n=nl;nl=nr;nr=n;}
    int lane=threadIdx.x&31;long long written=0;
    for(long long tile=0;tile<nl;tile+=32){
        long long k=tile+lane;bool found=false;int tid=0,value=0;
        if(k<nl){tid=lt[k];long long lo=0,hi=nr;
            while(lo<hi){long long m=(lo+hi)/2;if(rt[m]<tid)lo=m+1;else hi=m;}
            found=lo<nr && rt[lo]==tid;if(found){int a=lv[k],b=rv[lo];value=a<b?a:b;}}
        unsigned int mask=__ballot_sync(0xffffffff,found);
        if(found)csr_store(output_tids,output_values,output+written+__popc(mask&((1u<<lane)-1u)),tid,value);
        written+=__popc(mask);
    }
}
extern "C" __global__ void csr_pair_count_warp(const unsigned long long* ptrs,const long long* lengths,const int* bases,
    int n,long long first,long long jobs,int parallel,long long* counts){
    int lane=threadIdx.x&31;long long job=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(job>=jobs)return;long long i=0,j=0;
    if(!lane)csr_pair_ids(first+job,n,i,j);
    i=__shfl_sync(0xffffffff,i,0);j=__shfl_sync(0xffffffff,j,0);
    if(bases[i]==bases[j]){if(!lane)counts[job]=0;return;}
    long long count=csr_warp_count((const int*)ptrs[2*i],lengths[i],(const int*)ptrs[2*j],lengths[j]);
    if(!lane)counts[job]=count;
}
extern "C" __global__ void csr_pair_write_warp(const unsigned long long* ptrs,const long long* lengths,const int* bases,
    int n,long long first,long long jobs,int parallel,const long long* offsets,int* output_tids,short* output_values){
    int lane=threadIdx.x&31;long long job=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(job>=jobs)return;long long i=0,j=0;
    if(!lane)csr_pair_ids(first+job,n,i,j);
    i=__shfl_sync(0xffffffff,i,0);j=__shfl_sync(0xffffffff,j,0);
    if(bases[i]==bases[j])return;
    csr_warp_write((const int*)ptrs[2*i],(const short*)ptrs[2*i+1],lengths[i],
                   (const int*)ptrs[2*j],(const short*)ptrs[2*j+1],lengths[j],output_tids,output_values,offsets[job]);
}
extern "C" __global__ void csr_job_count_warp(const unsigned long long* ptrs,const long long* sizes,long long jobs,int parallel,long long* counts){
    int lane=threadIdx.x&31;long long job=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;if(job>=jobs)return;
    long long count=csr_warp_count((const int*)ptrs[4*job],sizes[4*job],(const int*)ptrs[4*job+2],sizes[4*job+1]);
    if(!lane)counts[job]=count;
}
extern "C" __global__ void csr_job_write_warp(const unsigned long long* ptrs,const long long* sizes,long long jobs,int parallel,
    const long long* offsets,int* output_tids,short* output_values){
    long long job=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;if(job>=jobs)return;
    csr_warp_write((const int*)ptrs[4*job],(const short*)ptrs[4*job+1],sizes[4*job],
                   (const int*)ptrs[4*job+2],(const short*)ptrs[4*job+3],sizes[4*job+1],output_tids,output_values,offsets[job]);
}
extern "C" __global__ void csr_supports_warp(const int* tids,const short* values,const long long* offsets,long long jobs,long long* supports){
    int lane=threadIdx.x&31;long long job=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;if(job>=jobs)return;
    long long support=0;
    for(long long k=offsets[job]+lane;k<offsets[job+1];k+=32){
#if CSR_PACKED
        support+=(unsigned int)tids[k]&15u;
#else
        support+=values[k];
#endif
    }
    for(int s=16;s;s/=2)support+=__shfl_down_sync(0xffffffff,support,s);
    if(!lane)supports[job]=support;
}

extern "C" __global__ void csr_region_count(const unsigned long long* ptrs,const long long* lengths,const int* lookup,
    const int* widths,int bases,int regions,long long first,long long jobs,int parallel,long long* counts){
    int lane=threadIdx.x&31;long long job=parallel?((long long)blockIdx.x*blockDim.x+threadIdx.x)/32:(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(job>=jobs)return;long long slot=first+job,slots=regions*regions,i=0,j=0;
    if(!parallel || !lane)csr_pair_ids(slot/slots,bases,i,j);
    if(parallel){i=__shfl_sync(0xffffffff,i,0);j=__shfl_sync(0xffffffff,j,0);}
    int a=(slot%slots)/regions,b=slot%regions;
    if(a>=widths[i] || b>=widths[j]){if(!parallel || !lane)counts[job]=0;return;}
    int left=lookup[i*regions+a],right=lookup[j*regions+b];
    long long count=parallel?csr_warp_count((const int*)ptrs[2*left],lengths[left],(const int*)ptrs[2*right],lengths[right]):
        csr_serial_merge((const int*)ptrs[2*left],0,lengths[left],(const int*)ptrs[2*right],0,lengths[right],0,0,0);
    if(!parallel || !lane)counts[job]=count;
}
extern "C" __global__ void csr_region_write(const unsigned long long* ptrs,const long long* lengths,const int* lookup,
    const int* widths,int bases,int regions,long long first,long long jobs,int parallel,
    const long long* offsets,int* output_tids,short* output_values){
    int lane=threadIdx.x&31;long long job=parallel?((long long)blockIdx.x*blockDim.x+threadIdx.x)/32:(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(job>=jobs)return;long long slot=first+job,slots=regions*regions,i=0,j=0;
    if(!parallel || !lane)csr_pair_ids(slot/slots,bases,i,j);
    if(parallel){i=__shfl_sync(0xffffffff,i,0);j=__shfl_sync(0xffffffff,j,0);}
    int a=(slot%slots)/regions,b=slot%regions;
    if(a>=widths[i] || b>=widths[j])return;
    int left=lookup[i*regions+a],right=lookup[j*regions+b];
    if(parallel)csr_warp_write((const int*)ptrs[2*left],(const short*)ptrs[2*left+1],lengths[left],
        (const int*)ptrs[2*right],(const short*)ptrs[2*right+1],lengths[right],output_tids,output_values,offsets[job]);
    else csr_serial_merge((const int*)ptrs[2*left],(const short*)ptrs[2*left+1],lengths[left],
        (const int*)ptrs[2*right],(const short*)ptrs[2*right+1],lengths[right],output_tids,output_values,offsets[job]);
}

"""


def counter_source(wide=False):


    return (CUDA_SOURCE.replace('typedef int support_counter_t;',
                                'typedef long long support_counter_t;') if wide else CUDA_SOURCE)


def candidate_arrays(prefixes, pair_matrix, neighbor_offsets, neighbors, term_bases, grouped):

    bound = 0
    for row in range(len(prefixes)):
        smallest = len(neighbors) + 1
        for term in prefixes[row]:
            size = neighbor_offsets[term+1] - neighbor_offsets[term]
            smallest = min(smallest, size)
        bound += smallest
    terms = np.empty(bound, dtype=np.int32)
    job_prefix = np.empty(bound, dtype=np.int64)
    job_right = np.empty(bound, dtype=np.int32)
    offsets = np.empty(bound + 1, dtype=np.int64)
    nc = nj = 0
    for row in range(len(prefixes)):
        smallest = len(neighbors) + 1
        chosen = prefixes[row, 0]
        for term in prefixes[row]:
            size = neighbor_offsets[term+1] - neighbor_offsets[term]
            if size < smallest:
                smallest, chosen = size, term
        previous_base = -1
        last_base = term_bases[prefixes[row, -1]]
        for k in range(neighbor_offsets[chosen], neighbor_offsets[chosen+1]):
            term = neighbors[k]
            base = term_bases[term]
            if base <= last_base:
                continue
            valid = True
            for old_term in prefixes[row]:
                if not pair_matrix[old_term, term]:
                    valid = False
                    break
            if not valid:
                continue
            if not grouped or base != previous_base:
                offsets[nj] = nc
                job_prefix[nj] = row
                job_right[nj] = base if grouped else term
                nj += 1
                previous_base = base
            terms[nc] = term
            nc += 1
    offsets[nj] = nc
    return terms[:nc], job_prefix[:nj], job_right[:nj], offsets[:nj+1]


NUMBA_IMPORT_ERROR = None
try:
    from numba import njit
except ImportError as exc:
    NUMBA_IMPORT_ERROR = str(exc)
else:
    candidate_arrays = njit(cache=True)(candidate_arrays)


def candidate_backend():
    return 'cuda'


class Matcher:

    def __init__(self, xp, threshold, results, scale):
        self.xp, self.threshold, self.results, self.scale = xp, threshold, results, scale
        self.jobs = self.candidates = 0
        self.max_jobs = 2048
        self.materialize_rows = 2_000_000
        if xp is not np:
            module = xp.RawModule(code=counter_source(wide=True))
            self.support_kernel = module.get_function('support_jobs')
            self.pair_kernel = module.get_function('pair_support')
            self.materialize_kernel = module.get_function('materialize')

    def descriptors(self, jobs):
        xp = self.xp
        ptrs, sizes = [], []
        for left, right, _ in jobs:
            ptrs.extend([left[0].data.ptr, left[1].data.ptr,
                         right[0].data.ptr, right[1].data.ptr])
            sizes.extend([len(left[0]), len(right[0]), left[1].shape[1], right[1].shape[1]])
        return xp.asarray(ptrs, dtype=xp.uint64), xp.asarray(sizes, dtype=xp.int64)

    def run(self, generator):
        retained = {}
        batch = []
        for job in generator:
            batch.append(job)
            if len(batch) == self.max_jobs:
                retained.update(self.evaluate(batch))
                batch = []
        if batch:
            retained.update(self.evaluate(batch))
        return retained

    def evaluate(self, jobs):
        xp = self.xp
        self.jobs += len(jobs)
        self.candidates += sum(len(choices) for _, _, choices in jobs)
        if xp is np:
            retained = {}
            for left, right, choices in jobs:
                li, ri = align(left[0], right[0], xp)
                for pattern, lc, rc in choices:
                    values = np.minimum(left[1][li, lc], right[1][ri, rc])
                    support = int(values.sum(dtype=np.int64))
                    if support >= self.threshold:
                        positive = values > 0
                        retained[pattern] = (left[0][li[positive]], values[positive, None])
                        self.results[pattern] = support / self.scale
            return retained
        ptrs, sizes = self.descriptors(jobs)
        offsets = np.cumsum([0] + [len(c) for _, _, c in jobs], dtype=np.int64)
        choices = [choice for _, _, cs in jobs for choice in cs]
        cols = xp.asarray([(lc, rc) for _, lc, rc in choices], dtype=xp.int32)
        sums = xp.empty(len(choices), dtype=xp.int64)
        max_choices = max(len(cs) for _, _, cs in jobs)
        self.support_kernel((len(jobs),), (128,),
                            (ptrs, sizes, xp.asarray(offsets), cols, sums),
                            shared_mem=max_choices * 128 * 8)
        host_sums = sums.get()
        frequent = []
        for index, (left, right, cs) in enumerate(jobs):
            for k, choice in enumerate(cs):
                support = int(host_sums[offsets[index] + k])
                if support >= self.threshold:
                    self.results[choice[0]] = support / self.scale
                    frequent.append((left, right, [choice]))

        retained, batch, rows = {}, [], 0
        for job in frequent:
            n = len(job[0][0])
            if batch and rows + n > self.materialize_rows:
                retained.update(self.materialize(batch))
                batch, rows = [], 0
            batch.append(job)
            rows += n
        if batch:
            retained.update(self.materialize(batch))
        return retained

    def pairs(self, bases, by_item, groups, singles, grouped):

        xp = self.xp
        count = len(bases)
        if count < 2:
            return {}
        widths = np.array([len(by_item[b]) for b in bases], dtype=np.int32)
        maxwidth = int(widths.max())
        slots = maxwidth * maxwidth
        if slots > 32:
            raise ValueError('CUDA supports at most five surviving regions per base item')
        term_ptrs = np.zeros((count, maxwidth, 2), dtype=np.uint64)
        term_lengths = np.zeros((count, maxwidth), dtype=np.int64)
        base_ptrs = np.zeros((count, 2), dtype=np.uint64)
        base_lengths = np.zeros(count, dtype=np.int64)
        for i, base in enumerate(bases):
            for col, term in enumerate(by_item[base]):
                t, v = singles[term]
                term_ptrs[i, col] = [t.data.ptr, v.data.ptr]
                term_lengths[i, col] = len(t)
            if grouped:
                t, v = groups[base]
                base_ptrs[i] = [t.data.ptr, v.data.ptr]
                base_lengths[i] = len(t)
        device_args = tuple(xp.asarray(a) for a in
                            (base_ptrs, base_lengths, term_ptrs, term_lengths, widths))
        pair_count = count * (count - 1) // 2
        candidate_count = (int(widths.sum()) ** 2 - int((widths.astype(np.int64) ** 2).sum())) // 2
        self.jobs += pair_count if grouped else candidate_count
        self.candidates += candidate_count
        retained = {}
        for first in range(0, pair_count, 262144):
            n = min(262144, pair_count - first)
            sums = xp.zeros(n * slots, dtype=xp.int64)
            self.pair_kernel((n if grouped else n * slots,), (128,),
                device_args + (np.int32(count), np.int32(maxwidth), np.int64(first),
                               np.int32(grouped), sums),
                shared_mem=(slots if grouped else 1) * 128 * 8)
            frequent_indices = xp.flatnonzero(sums >= self.threshold)
            host_indices = frequent_indices.get()
            host_supports = sums[frequent_indices].get()
            frequent, rows = [], 0
            for k, support in zip(host_indices, host_supports):
                pair, slot = first + int(k) // slots, int(k) % slots
                i = int(((2 * count - 1) - math.sqrt((2 * count - 1)**2 - 8 * pair)) / 2)
                while i * (2 * count - i - 1) // 2 > pair:
                    i -= 1
                while (i + 1) * (2 * count - i - 2) // 2 <= pair:
                    i += 1
                j = i + 1 + pair - i * (2 * count - i - 1) // 2
                lc, rc = divmod(slot, maxwidth)
                a, b = by_item[bases[i]][lc], by_item[bases[j]][rc]
                pattern = (a, b)
                self.results[pattern] = int(support) / self.scale
                if grouped:
                    left, right = groups[bases[i]], groups[bases[j]]
                else:
                    left, right, lc, rc = singles[a], singles[b], 0, 0
                if frequent and rows + len(left[0]) > self.materialize_rows:
                    retained.update(self.materialize(frequent))
                    frequent, rows = [], 0
                frequent.append((left, right, [(pattern, lc, rc)]))
                rows += len(left[0])
            if frequent:
                retained.update(self.materialize(frequent))
        return retained

    def extensions(self, current, terms, term_ids, term_bases, term_columns,
                   pair_matrix, neighbor_offsets, neighbors, bases, groups, singles, grouped):

        xp = self.xp
        patterns = list(current)
        occurrences = list(current.values())
        prefix_ids = np.array([[term_ids[t] for t in pattern] for pattern in patterns], dtype=np.int32)
        candidates, parents, right_indices, offsets = candidate_arrays(
            prefix_ids, pair_matrix, neighbor_offsets, neighbors, term_bases, grouped)
        self.candidates += len(candidates)
        self.jobs += len(parents)
        if not len(parents):
            return {}
        rights = [groups[b] for b in bases] if grouped else [singles[t] for t in terms]
        left_ptrs = np.array([[t.data.ptr, v.data.ptr] for t, v in occurrences], dtype=np.uint64)
        right_ptrs = np.array([[t.data.ptr, v.data.ptr] for t, v in rights], dtype=np.uint64)
        left_lengths = np.array([len(t) for t, _ in occurrences], dtype=np.int64)
        right_lengths = np.array([len(t) for t, _ in rights], dtype=np.int64)
        right_widths = np.array([v.shape[1] for _, v in rights], dtype=np.int64)
        if grouped:
            singleton_ptrs = np.array([[singles[t][0].data.ptr, singles[t][1].data.ptr] for t in terms], dtype=np.uint64)
            singleton_lengths = np.array([len(singles[t][0]) for t in terms], dtype=np.int64)
        retained = {}
        for first in range(0, len(parents), 32768):
            end = min(first + 32768, len(parents))
            cfirst, cend = int(offsets[first]), int(offsets[end])
            pi, ri = parents[first:end], right_indices[first:end]
            ptrs = np.column_stack((left_ptrs[pi], right_ptrs[ri]))
            sizes = np.column_stack((left_lengths[pi], right_lengths[ri],
                                     np.ones(len(pi), dtype=np.int64), right_widths[ri]))
            cols = np.zeros((cend-cfirst, 2), dtype=np.int32)
            if grouped:
                cols[:, 1] = term_columns[candidates[cfirst:cend]]
            local_offsets = offsets[first:end+1] - cfirst
            if grouped:
                one_region = np.diff(local_offsets) == 1
                only_candidates = local_offsets[:-1][one_region]
                only_terms = candidates[cfirst + only_candidates]
                ptrs[one_region, 2:] = singleton_ptrs[only_terms]
                sizes[one_region, 1] = singleton_lengths[only_terms]
                sizes[one_region, 3] = 1
                cols[only_candidates, 1] = 0
            sums = xp.empty(cend-cfirst, dtype=xp.int64)
            self.support_kernel((end-first,), (128,),
                (xp.asarray(ptrs), xp.asarray(sizes), xp.asarray(local_offsets), xp.asarray(cols), sums),
                shared_mem=int(np.diff(local_offsets).max()) * 128 * 8)
            host = sums.get()
            indices = np.flatnonzero(host >= self.threshold) + cfirst
            job_indices = np.searchsorted(offsets, indices, side='right') - 1
            frequent, rows = [], 0
            for c, j in zip(indices, job_indices):
                parent = int(parents[j])
                term_id = int(candidates[c])
                term = terms[term_id]
                pattern = patterns[parent] + (term,)
                self.results[pattern] = int(host[c-cfirst]) / self.scale
                singleton = grouped and offsets[j+1] - offsets[j] == 1
                left = occurrences[parent]
                right = singles[term] if singleton else rights[int(right_indices[j])]
                if frequent and rows + min(len(left[0]), len(right[0])) > self.materialize_rows:
                    retained.update(self.materialize(frequent))
                    frequent, rows = [], 0
                col = int(term_columns[term_id]) if grouped and not singleton else 0
                frequent.append((left, right, [(pattern, 0, col)]))
                rows += min(len(left[0]), len(right[0]))
            if frequent:
                retained.update(self.materialize(frequent))
        return retained

    def materialize(self, jobs):
        xp = self.xp
        jobs = [(right, left, [(cs[0][0], cs[0][2], cs[0][1])])
                if len(left[0]) > len(right[0]) else (left, right, cs)
                for left, right, cs in jobs]
        ptrs, sizes = self.descriptors(jobs)
        offsets = np.cumsum([0] + [len(left[0]) for left, _, _ in jobs], dtype=np.int64)
        cols = xp.asarray([(cs[0][1], cs[0][2]) for _, _, cs in jobs], dtype=xp.int32)
        tids = xp.empty(int(offsets[-1]), dtype=xp.int32)
        values = xp.empty(int(offsets[-1]), dtype=xp.int16)
        self.materialize_kernel((len(jobs),), (128,),
                                (ptrs, sizes, cols, xp.asarray(offsets), tids, values))
        positive = values > 0
        cumulative = xp.empty(len(values) + 1, dtype=xp.int64)
        cumulative[0] = 0
        xp.cumsum(positive, dtype=xp.int64, out=cumulative[1:])
        bounds = cumulative[xp.asarray(offsets)].get()
        compact_tids, compact_values = tids[positive], values[positive]
        return {cs[0][0]: (compact_tids[bounds[i]:bounds[i+1]],
                            compact_values[bounds[i]:bounds[i+1], None])
                for i, (_, _, cs) in enumerate(jobs)}


class PatternResults(Mapping):

    def __init__(self, terms, levels, scale):
        self.terms, self.levels, self.scale = terms, levels, scale
        self._index = None

    def __len__(self):
        return sum(len(supports) for _, supports in self.levels)

    def items(self):
        for patterns, supports in self.levels:
            for row, support in zip(patterns, supports):
                yield tuple(sorted(self.terms[t] for t in row)), int(support) / self.scale

    def __iter__(self):
        for pattern, _ in self.items():
            yield pattern

    def __getitem__(self, pattern):
        if self._index is None:
            self._index = dict(self.items())
        return self._index[pattern]


class OccurrencePool:

    def __init__(self, patterns, supports, tids, values, counts):
        self.patterns, self.supports = patterns, supports
        self.tids, self.values = tids, values
        self.lengths = np.asarray(counts, dtype=np.int64)
        self.offsets = np.r_[np.int64(0), np.cumsum(self.lengths, dtype=np.int64)]

    def pointers(self):
        offsets = self.offsets[:-1].astype(np.uint64)
        return np.column_stack((np.uint64(self.tids.data.ptr) + offsets * 4,
                                np.uint64(self.values.data.ptr) + offsets * 2))


def occurrence_table(occurrences):
    return (np.array([[t.data.ptr, v.data.ptr] for t, v in occurrences], dtype=np.uint64).reshape(-1, 2),
            np.array([len(t) for t, _ in occurrences], dtype=np.int64),
            np.array([v.shape[1] for _, v in occurrences], dtype=np.int64))


NAIVE_KERNEL = 'auto'


class PoolMatcher:

    def __init__(self, xp, threshold, bases, by_item, singles, groups, grouped, wide=False, transactions=0, membership_scale=10):
        self.xp, self.threshold, self.grouped = xp, threshold, grouped
        self.naive_kernel = NAIVE_KERNEL
        self.bases, self.by_item = bases, by_item
        self.terms = [term for base in bases for term in by_item[base]]
        self.term_ids = {term: i for i, term in enumerate(self.terms)}
        base_ids = {base: i for i, base in enumerate(bases)}
        self.term_bases = np.array([base_ids[t[0]] for t in self.terms], dtype=np.int32)
        self.term_columns = np.array([by_item[t[0]].index(t) for t in self.terms], dtype=np.int32)
        self.single_table = occurrence_table([singles[t] for t in self.terms])
        self.mean_length=float(self.single_table[1].mean()) if self.terms else 0
        self.group_threads=64 if self.mean_length<512 else 128
        self.use_pivots=bool(grouped and transactions and self.mean_length/transactions<.1)

        if self.naive_kernel=='auto':
            mean=float(self.single_table[1].mean()) if len(self.terms) else 0
            density=mean/max(transactions,1)
            if len(self.terms)<64:self.naive_kernel='serial'
            elif mean<=512:self.naive_kernel='csr_serial'
            elif mean>=4096 and 1/128<=density<.1:self.naive_kernel='csr_parallel'
            else:self.naive_kernel='atomic'

        self.group_table = occurrence_table([groups[b] for b in bases]) if grouped else None
        self.pack_csr=bool(membership_scale<=15 and 0<transactions<=1<<28)
        source=counter_source(wide)
        max_regions=max((len(by_item[b]) for b in bases),default=1)
        source=source.replace('#define MAX_PAIR_COUNTERS 32',f'#define MAX_PAIR_COUNTERS {max_regions*max_regions}')
        source=source.replace('#define MAX_EXTENSION_COUNTERS 32',f'#define MAX_EXTENSION_COUNTERS {max_regions}')
        if self.pack_csr:source=source.replace('#define CSR_PACKED 0','#define CSR_PACKED 1')
        if self.use_pivots:source=source.replace('#define SPARSE_PIVOT 0','#define SPARSE_PIVOT 1')
        module = xp.RawModule(code=source)
        self.counter_bytes = 8 if wide else 4
        self.support_kernel = module.get_function('support_jobs' if grouped else 'flat_support_jobs')
        self.flat_pair_kernel = module.get_function('flat_pair_serial' if self.naive_kernel=='serial' else 'flat_pair_support')
        self.serial_support_kernel = module.get_function('flat_support_serial')
        self.count_kernel = module.get_function('extension_counts')
        self.fill_kernel = module.get_function('extension_fill')
        self.descriptor_kernel = module.get_function('extension_descriptors')
        self.csr_pair_count_kernel=module.get_function('csr_pair_count_warp' if self.naive_kernel=='csr_parallel' else 'csr_pair_count')
        self.csr_pair_write_kernel=module.get_function('csr_pair_write_warp' if self.naive_kernel=='csr_parallel' else 'csr_pair_write')
        self.csr_job_count_kernel=module.get_function('csr_job_count_warp' if self.naive_kernel=='csr_parallel' else 'csr_job_count')
        self.csr_job_write_kernel=module.get_function('csr_job_write_warp' if self.naive_kernel=='csr_parallel' else 'csr_job_write')
        self.csr_support_kernel=module.get_function('csr_supports_warp' if self.naive_kernel=='csr_parallel' else 'csr_supports')
        self.csr_select_kernel=module.get_function('csr_select')
        self.csr_region_count_kernel=module.get_function('csr_region_count')
        self.csr_region_write_kernel=module.get_function('csr_region_write')
        self.device_graph = None
        self.max_regions = max((len(by_item[b]) for b in bases), default=1)
        self.pair_kernel = module.get_function('pair_support')
        self.materialize_kernel = module.get_function('materialize')
        self.jobs = self.candidates = 0
        self.row_budget = 64_000_000

    def make_pool(self, parts, length):
        xp = self.xp
        if not parts:
            return OccurrencePool(np.empty((0, length), dtype=np.int32), np.empty(0, dtype=np.int64),
                                  xp.empty(0, dtype=xp.int32), xp.empty(0, dtype=xp.int16), [])
        return OccurrencePool(np.concatenate([p.patterns for p in parts]),
                              np.concatenate([p.supports for p in parts]),
                              xp.concatenate([p.tids for p in parts]),
                              xp.concatenate([p.values for p in parts]),
                              np.concatenate([p.lengths for p in parts]))

    def materialize(self, patterns, supports, pointers, sizes, columns):

        xp = self.xp
        if not len(patterns):
            return self.make_pool([], patterns.shape[1])
        pointers, sizes, columns = pointers.copy(), sizes.copy(), columns.copy()
        swap = sizes[:, 0] > sizes[:, 1]
        pointers[swap] = pointers[swap][:, [2, 3, 0, 1]]
        sizes[swap] = sizes[swap][:, [1, 0, 3, 2]]
        columns[swap] = columns[swap][:, [1, 0]]
        dense_offsets = np.r_[np.int64(0), np.cumsum(sizes[:, 0], dtype=np.int64)]
        chunks_t, chunks_v, chunks_n = [], [], []
        first = 0
        while first < len(patterns):
            end = max(first + 1, int(np.searchsorted(dense_offsets,
                      dense_offsets[first] + self.row_budget, side='right')) - 1)
            end = min(end, len(patterns))
            offsets = dense_offsets[first:end+1] - dense_offsets[first]
            rows = int(offsets[-1])
            tids, values = xp.empty(rows, dtype=xp.int32), xp.empty(rows, dtype=xp.int16)
            self.materialize_kernel((end-first,), (128,),
                (xp.asarray(pointers[first:end]), xp.asarray(sizes[first:end]),
                 xp.asarray(columns[first:end]), xp.asarray(offsets), tids, values))
            positive = values > 0

            cumulative = xp.empty(rows + 1, dtype=xp.int32)
            cumulative[0] = 0
            xp.cumsum(positive, dtype=xp.int32, out=cumulative[1:])
            bounds = cumulative[xp.asarray(offsets)].get()
            chunks_n.append(np.diff(bounds).astype(np.int64))
            chunks_t.append(tids[positive])
            chunks_v.append(values[positive])
            first = end
        return OccurrencePool(patterns, supports, xp.concatenate(chunks_t),
                              xp.concatenate(chunks_v), np.concatenate(chunks_n))

    def pairs(self):
        if not self.grouped:
            return self.flat_pairs()
        xp = self.xp
        n = len(self.bases)
        if n < 2:
            return self.make_pool([], 2)
        widths = np.array([len(self.by_item[b]) for b in self.bases], dtype=np.int32)
        maxwidth = int(widths.max())
        slots = maxwidth * maxwidth
        if slots > 32:
            raise ValueError('CUDA supports at most five surviving regions per base item')
        term_ptrs = np.zeros((n, maxwidth, 2), dtype=np.uint64)
        term_lengths = np.zeros((n, maxwidth), dtype=np.int64)
        term_lookup = np.zeros((n, maxwidth), dtype=np.int32)
        for t in range(len(self.terms)):
            b, r = self.term_bases[t], self.term_columns[t]
            term_ptrs[b, r] = self.single_table[0][t]
            term_lengths[b, r] = self.single_table[1][t]
            term_lookup[b, r] = t
        base_ptrs = self.group_table[0] if self.grouped else np.zeros((n, 2), dtype=np.uint64)
        base_lengths = self.group_table[1] if self.grouped else np.zeros(n, dtype=np.int64)
        args = tuple(xp.asarray(a) for a in (base_ptrs, base_lengths, term_ptrs, term_lengths, widths))
        npairs = n * (n-1) // 2
        ncandidates = (int(widths.sum())**2 - int((widths.astype(np.int64)**2).sum())) // 2
        self.jobs += npairs if self.grouped else ncandidates
        self.candidates += ncandidates
        parts = []
        for first in range(0, npairs, 1_048_576):
            count = min(1_048_576, npairs-first)
            sums = xp.zeros(count * slots, dtype=xp.int64)
            self.pair_kernel((count if self.grouped else count*slots,), (self.group_threads,),
                args + (np.int32(n), np.int32(maxwidth), np.int64(first), np.int32(self.grouped), sums),
                shared_mem=(slots if self.grouped else 1)*(self.group_threads//32)*self.counter_bytes)
            indices = xp.flatnonzero(sums >= self.threshold)
            supports = sums[indices].get()
            selected = indices.get()
            if not len(selected):
                continue
            pair, slot = first + selected//slots, selected%slots
            i = (((2*n-1) - np.sqrt((2*n-1)**2 - 8*pair))/2).astype(np.int64)
            while np.any(i*(2*n-i-1)//2 > pair):
                i[i*(2*n-i-1)//2 > pair] -= 1
            while np.any((i+1)*(2*n-i-2)//2 <= pair):
                i[(i+1)*(2*n-i-2)//2 <= pair] += 1
            j = i+1+pair-i*(2*n-i-1)//2
            lc, rc = slot//maxwidth, slot%maxwidth
            a, b = term_lookup[i, lc], term_lookup[j, rc]
            patterns = np.column_stack((a, b))
            if self.grouped:
                pointers = np.column_stack((base_ptrs[i], base_ptrs[j]))
                sizes = np.column_stack((base_lengths[i], base_lengths[j], widths[i], widths[j])).astype(np.int64)
                columns = np.column_stack((lc, rc)).astype(np.int32)
            else:
                pointers = np.column_stack((self.single_table[0][a], self.single_table[0][b]))
                sizes = np.column_stack((self.single_table[1][a], self.single_table[1][b],
                                         np.ones(len(a), dtype=np.int64), np.ones(len(a), dtype=np.int64)))
                columns = np.zeros((len(a), 2), dtype=np.int32)
            parts.append(self.materialize(patterns, supports, pointers, sizes, columns))
        return self.make_pool(parts, 2)

    def extensions_host(self, current, pair_matrix, neighbor_offsets, neighbors):
        xp = self.xp
        terms, parents, rights, offsets = candidate_arrays(current.patterns, pair_matrix,
            neighbor_offsets, neighbors, self.term_bases, self.grouped)
        self.candidates += len(terms)
        self.jobs += len(parents)
        table = self.group_table if self.grouped else self.single_table
        current_ptrs = current.pointers()
        parts = []
        for first in range(0, len(parents), 1_000_000):
            end = min(first+1_000_000, len(parents))
            cfirst, cend = int(offsets[first]), int(offsets[end])
            pi, ri = parents[first:end], rights[first:end]
            pointers = np.column_stack((current_ptrs[pi], table[0][ri]))
            sizes = np.column_stack((current.lengths[pi], table[1][ri],
                                      np.ones(len(pi), dtype=np.int64), table[2][ri]))
            columns = np.zeros((cend-cfirst, 2), dtype=np.int32)
            local_offsets = offsets[first:end+1] - cfirst
            one = np.diff(local_offsets) == 1
            if self.grouped:
                columns[:, 1] = self.term_columns[terms[cfirst:cend]]
                positions = local_offsets[:-1][one]
                singleton_terms = terms[cfirst+positions]
                pointers[one, 2:] = self.single_table[0][singleton_terms]
                sizes[one, 1] = self.single_table[1][singleton_terms]
                sizes[one, 3] = 1
                columns[positions, 1] = 0
            sums = xp.empty(cend-cfirst, dtype=xp.int64)
            self.support_kernel((end-first,), (128,),
                (xp.asarray(pointers), xp.asarray(sizes), xp.asarray(local_offsets), xp.asarray(columns), sums),
                shared_mem=int(np.diff(local_offsets).max())*128*8)
            host = sums.get()
            selected = np.flatnonzero(host >= self.threshold)
            if not len(selected):
                continue
            candidates = selected + cfirst
            jobs = np.searchsorted(offsets, candidates, side='right') - 1
            parent, right = parents[jobs], rights[jobs]
            child = terms[candidates]
            patterns = np.column_stack((current.patterns[parent], child)).astype(np.int32)
            fpointers = np.column_stack((current_ptrs[parent], table[0][right]))
            fsizes = np.column_stack((current.lengths[parent], table[1][right],
                                      np.ones(len(parent), dtype=np.int64), table[2][right]))
            fcolumns = np.zeros((len(child), 2), dtype=np.int32)
            if self.grouped:
                fcolumns[:, 1] = self.term_columns[child]
                singleton = offsets[jobs+1] - offsets[jobs] == 1
                fpointers[singleton, 2:] = self.single_table[0][child[singleton]]
                fsizes[singleton, 1] = self.single_table[1][child[singleton]]
                fsizes[singleton, 3] = 1
                fcolumns[singleton, 1] = 0
            parts.append(self.materialize(patterns, host[selected], fpointers, fsizes, fcolumns))
        return self.make_pool(parts, current.patterns.shape[1]+1)

    def flat_pairs(self):
        xp = self.xp
        n = len(self.terms)
        if n < 2:
            return self.make_pool([], 2)
        same_base = np.bincount(self.term_bases, minlength=len(self.bases)).astype(np.int64)
        self.candidates += n*(n-1)//2 - int((same_base*(same_base-1)//2).sum())
        self.jobs += self.candidates
        dp, dn, db = (xp.asarray(self.single_table[0]), xp.asarray(self.single_table[1]), xp.asarray(self.term_bases))
        parts = []
        region_grid=self.naive_kernel=='csr_serial'
        if region_grid:
            widths=np.array([len(self.by_item[b]) for b in self.bases],dtype=np.int32)
            maxregions=int(widths.max())
            lookup=np.full((len(self.bases),maxregions),-1,dtype=np.int32)
            lookup[self.term_bases,self.term_columns]=np.arange(n,dtype=np.int32)
            dl,dw=xp.asarray(lookup),xp.asarray(widths)
            total_pairs=len(self.bases)*(len(self.bases)-1)//2*maxregions*maxregions
        else:
            total_pairs=n*(n-1)//2
        for first in range(0, total_pairs, 1_048_576):
            count = min(1_048_576, total_pairs-first)
            sums = xp.zeros(count, dtype=xp.int64)
            bundle=(self.csr_region_pairs(dp,dn,dl,dw,len(self.bases),maxregions,first,count) if region_grid
                    else self.csr_pairs(dp,dn,db,n,first,count) if self.naive_kernel=='csr_parallel' else None)
            if region_grid and bundle is None:
                raise MemoryError('Candidate intersection workspace exceeds 4 GiB; reduce the candidate batch size')
            if bundle is not None:
                sums=bundle[0]
            elif self.naive_kernel=='serial':
                self.flat_pair_kernel(((count+127)//128,),(128,),
                    (dp,dn,db,np.int32(n),np.int64(first),np.int64(count),sums))
            else:
                self.flat_pair_kernel((count,), (256,), (dp, dn, db, np.int32(n), np.int64(first), sums))
            selected_device = xp.flatnonzero(sums >= self.threshold)
            supports, selected = sums[selected_device].get(), selected_device.get()
            if not len(selected):
                continue
            slot=first+selected
            pair=slot//(maxregions*maxregions) if region_grid else slot
            size=len(self.bases) if region_grid else n
            i = (((2*size-1)-np.sqrt((2*size-1)**2-8*pair))/2).astype(np.int64)
            while np.any(i*(2*size-i-1)//2 > pair):
                i[i*(2*size-i-1)//2 > pair] -= 1
            while np.any((i+1)*(2*size-i-2)//2 <= pair):
                i[(i+1)*(2*size-i-2)//2 <= pair] += 1
            j = i+1+pair-i*(2*size-i-1)//2
            if region_grid:
                i,j=lookup[i,(slot%(maxregions*maxregions))//maxregions],lookup[j,slot%maxregions]
            patterns = np.column_stack((i,j)).astype(np.int32)
            if bundle is not None:
                parts.append(self.select_csr(patterns,supports,selected_device,bundle))
                continue
            pointers = np.column_stack((self.single_table[0][i], self.single_table[0][j]))
            sizes = np.column_stack((self.single_table[1][i], self.single_table[1][j],
                                     np.ones(len(i), dtype=np.int64), np.ones(len(i), dtype=np.int64)))
            parts.append(self.materialize(patterns, supports, pointers, sizes, np.zeros((len(i),2),dtype=np.int32)))
        return self.make_pool(parts,2)

    def extensions(self, current, pair_matrix, neighbor_offsets, neighbors):
        xp = self.xp
        if self.device_graph is None:
            self.device_graph = tuple(xp.asarray(a) for a in
                (pair_matrix,neighbor_offsets,neighbors,self.term_bases,self.term_columns))
            table = self.group_table if self.grouped else self.single_table
            self.device_right = tuple(xp.asarray(a) for a in table)
            self.device_single = tuple(xp.asarray(a) for a in self.single_table[:2])
        pair_gpu, noff, adj, bases_gpu, region_gpu = self.device_graph
        patterns_gpu = xp.asarray(current.patterns)
        rows, length = current.patterns.shape
        counts = xp.empty((2,rows),dtype=xp.int64)
        args = (patterns_gpu,np.int32(length),np.int64(rows),np.int32(len(self.terms)),
                pair_gpu,noff,adj,bases_gpu,np.int32(self.grouped))
        grid=((rows+127)//128,)
        self.count_kernel(grid,(128,),args+(counts[0],counts[1]))
        coff,joff=xp.empty(rows+1,dtype=xp.int64),xp.empty(rows+1,dtype=xp.int64)
        coff[0]=0;joff[0]=0
        xp.cumsum(counts[0],dtype=xp.int64,out=coff[1:])
        xp.cumsum(counts[1],dtype=xp.int64,out=joff[1:])
        totals=xp.stack((coff[-1],joff[-1])).get()
        nc,nj=int(totals[0]),int(totals[1])
        self.candidates+=nc;self.jobs+=nj
        if not nj:
            return self.make_pool([],length+1)
        terms=xp.empty(nc,dtype=xp.int32);parents=xp.empty(nj,dtype=xp.int64)
        rights=xp.empty(nj,dtype=xp.int32);offsets=xp.empty(nj+1,dtype=xp.int64)
        self.fill_kernel(grid,(128,),args+(coff,joff,terms,parents,rights,offsets))
        offsets[-1]=nc
        pointers=xp.empty((nj,4),dtype=xp.uint64);sizes=xp.empty((nj,4),dtype=xp.int64)
        columns=xp.empty((nc,2),dtype=xp.int32)
        rptr,rlength,rwidth=self.device_right;sptr,slength=self.device_single
        self.descriptor_kernel(((nj+127)//128,),(128,),
            (np.int64(nj),np.int32(self.grouped),terms,parents,rights,offsets,xp.asarray(current.offsets),
             np.uint64(current.tids.data.ptr),np.uint64(current.values.data.ptr),rptr,rlength,rwidth,
             sptr,slength,region_gpu,pointers,sizes,columns))
        sums=xp.empty(nc,dtype=xp.int64)
        bundle=self.csr_jobs(pointers,sizes,nj) if not self.grouped and self.naive_kernel.startswith('csr_') else None
        if bundle is not None:
            sums=bundle[0]
        elif not self.grouped and self.naive_kernel=='serial':
            self.serial_support_kernel(((nj+127)//128,),(128,),
                (pointers,sizes,offsets,sums,np.int64(nj)))
        else:
            self.support_kernel((nj,),(self.group_threads if self.grouped else 256,),(pointers,sizes,offsets,columns,sums),shared_mem=self.max_regions*(self.group_threads//32)*self.counter_bytes if self.grouped else 0)
        selected=xp.flatnonzero(sums>=self.threshold)
        if not len(selected):
            return self.make_pool([],length+1)
        jobs=xp.searchsorted(offsets,selected,side='right')-1
        new_patterns=xp.column_stack((patterns_gpu[parents[jobs]],terms[selected])).get()
        supports=sums[selected].get()
        if bundle is not None:
            return self.select_csr(new_patterns,supports,selected,bundle)
        return self.materialize(new_patterns,supports,pointers[jobs].get(),sizes[jobs].get(),columns[selected].get())


    def finish_csr(self,counts,writer):
        xp=self.xp;n=len(counts)
        offsets=xp.empty(n+1,dtype=xp.int64);offsets[0]=0
        xp.cumsum(counts,dtype=xp.int64,out=offsets[1:])
        rows=int(offsets[-1].get())
        if rows*(4 if self.pack_csr else 6)>4*1024**3:
            return None
        tids=xp.empty(rows,dtype=xp.int32);values=xp.empty(0 if self.pack_csr else rows,dtype=xp.int16)
        writer(offsets,tids,values)
        sums=xp.empty(n,dtype=xp.int64)
        self.csr_support_kernel((((n*32 if self.naive_kernel=='csr_parallel' else n)+127)//128,),(128,),(tids,values,offsets,np.int64(n),sums))
        return sums,counts,offsets,tids,values

    def csr_region_pairs(self,pointers,lengths,lookup,widths,bases,regions,first,jobs):
        xp=self.xp;parallel=self.naive_kernel=='csr_parallel'
        grid=((jobs*32+127)//128,) if parallel else ((jobs+127)//128,)
        args=(pointers,lengths,lookup,widths,np.int32(bases),np.int32(regions),np.int64(first),np.int64(jobs),np.int32(parallel))
        counts=xp.empty(jobs,dtype=xp.int64)
        self.csr_region_count_kernel(grid,(128,),args+(counts,))
        return self.finish_csr(counts,lambda offsets,tids,values:
            self.csr_region_write_kernel(grid,(128,),args+(offsets,tids,values)))

    def csr_pairs(self,pointers,lengths,bases,n,first,jobs):
        xp=self.xp;parallel=self.naive_kernel=='csr_parallel'
        grid=((jobs*32+127)//128,) if parallel else ((jobs+127)//128,)
        threads=(128,)
        args=(pointers,lengths,bases,np.int32(n),np.int64(first),np.int64(jobs),np.int32(parallel))
        counts=xp.empty(jobs,dtype=xp.int64)
        self.csr_pair_count_kernel(grid,threads,args+(counts,))
        return self.finish_csr(counts,lambda offsets,tids,values:
            self.csr_pair_write_kernel(grid,threads,args+(offsets,tids,values)))

    def csr_jobs(self,pointers,sizes,jobs):
        xp=self.xp;parallel=self.naive_kernel=='csr_parallel'
        grid=((jobs*32+127)//128,) if parallel else ((jobs+127)//128,)
        threads=(128,)
        args=(pointers,sizes,np.int64(jobs),np.int32(parallel))
        counts=xp.empty(jobs,dtype=xp.int64)
        self.csr_job_count_kernel(grid,threads,args+(counts,))
        return self.finish_csr(counts,lambda offsets,tids,values:
            self.csr_job_write_kernel(grid,threads,args+(offsets,tids,values)))

    def select_csr(self,patterns,supports,selected,bundle):
        xp=self.xp
        _,counts,offsets,tids,values=bundle
        host_counts=counts[selected].get()
        output_offsets=np.r_[np.int64(0),np.cumsum(host_counts,dtype=np.int64)]
        output_tids=xp.empty(int(output_offsets[-1]),dtype=xp.int32)
        output_values=xp.empty(int(output_offsets[-1]),dtype=xp.int16)
        self.csr_select_kernel((len(selected),),(256,),
            (tids,values,offsets,selected,xp.asarray(output_offsets),output_tids,output_values))
        return OccurrencePool(patterns,supports,output_tids,output_values,host_counts)


def pooled_levels(xp, threshold, bases, by_item, singles, groups, initial_results, scale, transactions=0):
    wide = math.ceil(max(initial_results.values(), default=0)*scale) > np.iinfo(np.int32).max//32
    matcher = PoolMatcher(xp, threshold, bases, by_item, singles, groups, grouped=bool(groups), wide=wide, transactions=transactions, membership_scale=scale)
    terms = matcher.terms
    f1 = np.arange(len(terms), dtype=np.int32).reshape(-1, 1)
    supports = np.rint(np.array([initial_results[(t,)] for t in terms])*scale).astype(np.int64)
    result_levels = [(f1, supports)]
    current = matcher.pairs()
    levels = [len(terms), len(current.supports)]
    if len(current.supports):
        result_levels.append((current.patterns, current.supports))
        pair_matrix = np.zeros((len(terms), len(terms)), dtype=np.bool_)
        a, b = current.patterns[:, 0], current.patterns[:, 1]
        pair_matrix[a, b] = True
        order = np.lexsort((b, a))
        neighbors = b[order].astype(np.int32)
        neighbor_offsets = np.r_[np.int64(0), np.cumsum(np.bincount(a, minlength=len(terms)), dtype=np.int64)]
    while len(current.supports):
        current = matcher.extensions(current, pair_matrix, neighbor_offsets, neighbors)
        if len(current.supports):
            levels.append(len(current.supports))
            result_levels.append((current.patterns, current.supports))
    return PatternResults(terms, result_levels, scale), {
        'levels': levels, 'candidates': matcher.candidates, 'list_matches': matcher.jobs,
        'occurrence_storage': 'pooled CSR', 'result_storage': 'term-ID matrices + exact supports',
        'support_accumulation': f'int{matcher.counter_bytes*8} block reduction' if matcher.grouped else ('CSR materialize then int64 sum' if matcher.naive_kernel.startswith('csr_') else ('int64 serial merge' if matcher.naive_kernel=='serial' else 'int64 global atomics')),
        'driving_list': 'shorter list' if matcher.grouped or matcher.naive_kernel=='csr_parallel' else 'prefix list',
        'naive_kernel': None if matcher.grouped else matcher.naive_kernel,
        'csr_encoding': 'packed uint32 TID/membership' if matcher.pack_csr else 'int32 TID + int16 membership',
        'shared_probe_pivots': matcher.use_pivots if matcher.grouped else False,
        'group_threads': matcher.group_threads if matcher.grouped else None}


def build_gpu_groups(xp, bases, by_item, singles, raw_groups, raw_layout):

    groups = {base: singles[by_item[base][0]] for base in bases if len(by_item[base]) == 1}
    selected = [base for base in bases if len(by_item[base]) > 1]
    if not selected:
        return groups
    raw_tids, raw_values, offsets_t, offsets_v, raw_widths, base_ids = raw_layout
    metadata, columns, shapes = [], [], []
    dense = 0
    colstride = max(len(by_item[base]) for base in selected)
    for base in selected:
        i = base_ids[base]
        n = int(offsets_t[i+1] - offsets_t[i])
        nr = len(by_item[base])
        metadata.append((offsets_t[i], offsets_v[i], raw_widths[i], n, nr, dense))
        cols = [raw_groups[base][2][term] for term in by_item[base]]
        columns.append(cols + [0] * (colstride-nr))
        shapes.append(nr)
        dense += n
    metadata = np.array(metadata, dtype=np.int64)
    columns = np.array(columns, dtype=np.int32)
    shape_array = np.array(shapes, dtype=np.int64)
    dense_offsets = np.r_[metadata[:, 5], np.int64(dense)]
    rt, rv = xp.asarray(raw_tids), xp.asarray(raw_values)
    dm, dc = xp.asarray(metadata), xp.asarray(columns)
    flags = xp.empty(dense, dtype=xp.uint8)
    module = xp.RawModule(code=CUDA_SOURCE)
    module.get_function('grouped_flags')((len(selected),), (128,),
                                        (rv, dm, dc, np.int32(colstride), flags))
    scan = xp.empty(dense+1, dtype=xp.int32)
    scan[0] = 0
    xp.cumsum(flags, dtype=xp.int32, out=scan[1:])
    bounds = scan[xp.asarray(dense_offsets)].get().astype(np.int64)
    counts = np.diff(bounds)
    value_offsets = np.r_[np.int64(0), np.cumsum(counts*shape_array, dtype=np.int64)]
    output_offsets = np.column_stack((bounds[:-1], value_offsets[:-1]))
    tids = xp.empty(int(bounds[-1]), dtype=xp.int32)
    values = xp.empty(int(value_offsets[-1]), dtype=xp.int16)
    module.get_function('grouped_copy')((len(selected),), (128,),
        (rt, rv, dm, dc, np.int32(colstride), flags, scan, xp.asarray(output_offsets), tids, values))
    for i, base in enumerate(selected):
        groups[base] = (tids[bounds[i]:bounds[i+1]],
                        values[value_offsets[i]:value_offsets[i+1]].reshape(int(counts[i]), shapes[i]))
    return groups


def unpack_region(code):
    letters=[]
    while code:
        letters.append(chr((code&31)-1+ord('A')));code>>=5
    return ''.join(reversed(letters))


def repeat_gpu_lists(xp,tids,values,counts,widths,repeat,transactions):
    counts=np.asarray(counts,dtype=np.int64);widths=np.asarray(widths,dtype=np.int64)
    offsets=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    value_offsets=np.r_[np.int64(0),np.cumsum(counts*widths,dtype=np.int64)]
    if repeat==1:
        return tids,values,offsets,value_offsets
    out_offsets=offsets*repeat;out_value_offsets=value_offsets*repeat
    out_t=xp.empty(int(out_offsets[-1]),dtype=xp.int32);out_v=xp.empty(int(out_value_offsets[-1]),dtype=xp.int16)
    kernel=xp.RawModule(code=CUDA_SOURCE).get_function('repeat_occurrences')
    kernel((len(counts),),(128,),(tids,values,xp.asarray(offsets),xp.asarray(value_offsets),xp.asarray(widths),
        np.int32(repeat),np.int32(transactions),xp.asarray(out_offsets),xp.asarray(out_value_offsets),out_t,out_v))
    return out_t,out_v,out_offsets,out_value_offsets


def csv_positions_gpu(xp,buf):
    if len(buf)<64*1024*1024:
        return xp.flatnonzero(buf==10),xp.flatnonzero(buf==58)
    blocks=(len(buf)+255)//256
    counts=xp.empty((blocks,2),dtype=xp.int32)
    module=xp.RawModule(code=CUDA_SOURCE)
    module.get_function('csv_position_counts')((blocks,),(256,),(buf,np.int64(len(buf)),counts))
    aoff=xp.cumsum(counts[:,0],dtype=xp.int32);boff=xp.cumsum(counts[:,1],dtype=xp.int32)
    totals=xp.stack((aoff[-1],boff[-1])).get()
    ends=xp.empty(int(totals[0]),dtype=xp.int64);colons=xp.empty(int(totals[1]),dtype=xp.int64)
    module.get_function('csv_position_write')((blocks,),(256,),(buf,np.int64(len(buf)),aoff,boff,ends,colons))
    return ends,colons


def byte_positions_gpu(xp,buf,target):
    if not len(buf):return xp.empty(0,dtype=xp.int64)
    prefix=xp.cumsum(buf==target,dtype=xp.int32)
    count=int(prefix[-1].get())
    output=xp.empty(count,dtype=xp.int64)
    xp.RawModule(code=CUDA_SOURCE).get_function('byte_positions')(((len(buf)+255)//256,),(256,),
        (buf,prefix,np.int64(len(buf)),np.int32(target),output))
    return output


def record_supports_gpu(xp,codes,values,terms,scale):
    wide=len(codes)*scale>np.iinfo(np.int32).max
    output=xp.zeros(terms,dtype=xp.int64 if wide else xp.int32)
    if len(codes):xp.RawModule(code=CUDA_SOURCE).get_function('record_supports')(
        ((len(codes)+255)//256,),(256,),(codes,values,np.int64(len(codes)),np.int32(wide),output))
    return output.astype(xp.int64,copy=False)


def mine_file_gpu(path,min_sup,xp,grouped=True,repeat=1,membership_scale=10,duplicate_policy='max',relative_support=False):

    from decimal import Decimal,ROUND_CEILING
    if not math.isfinite(min_sup) or min_sup<=0 or not 1<=membership_scale<=32767 or repeat<1:
        raise ValueError('Invalid support, membership scale or repetition count')
    threshold=int((Decimal(str(min_sup))*membership_scale).to_integral_value(rounding=ROUND_CEILING))
    import mmap,os
    with open(path,'rb') as f:
        input_bytes=os.fstat(f.fileno()).st_size
        if not input_bytes:
            return {},{'levels':[0],'candidates':0,'list_matches':0,'transactions':0,'base_items':0,'input_bytes':0}
        if not grouped:
            raw=f.read();final_byte=raw[-1]
            host=np.frombuffer(raw,dtype=np.uint8)
            buf=xp.empty(host.size,dtype=xp.uint8)
            xp.cuda.runtime.memcpy(buf.data.ptr,host.ctypes.data,host.nbytes,1)
            xp.cuda.Stream.null.synchronize()
        else:

            with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped:
                final_byte=mapped[-1]
                host=np.frombuffer(mapped,dtype=np.uint8)
                buf=xp.empty(host.size,dtype=xp.uint8)
                xp.cuda.runtime.memcpy(buf.data.ptr,host.ctypes.data,host.nbytes,1)
                xp.cuda.Stream.null.synchronize()
                del host
    if final_byte!=10:buf=xp.concatenate((buf,xp.array([10],dtype=xp.uint8)))
    ends,colons=csv_positions_gpu(xp,buf) if grouped else (xp.flatnonzero(buf==10),xp.flatnonzero(buf==58))
    transactions=len(colons)
    if relative_support:
        threshold=int((Decimal(str(min_sup))*transactions*repeat*membership_scale).to_integral_value(rounding=ROUND_CEILING))
    if transactions*repeat>2147483647:
        raise ValueError('Repeated transaction IDs exceed int32')
    if not transactions:
        return {},{'levels':[0],'candidates':0,'list_matches':0,'transactions':0,'base_items':0,'input_bytes':input_bytes}
    line=xp.searchsorted(ends,colons)
    starts=xp.where(line>0,ends[xp.maximum(line-1,0)]+1,0)
    line_ends=ends[line]
    module=xp.RawModule(code=CUDA_SOURCE)
    counts=xp.empty(transactions,dtype=xp.int64)
    grid=((transactions+127)//128,)
    module.get_function('csv_counts')(grid,(128,),(buf,starts,colons,np.int64(transactions),counts))
    offsets=xp.empty(transactions+1,dtype=xp.int64);offsets[0]=0
    xp.cumsum(counts,dtype=xp.int64,out=offsets[1:])
    records=int(offsets[-1].get())
    tids=xp.empty(records,dtype=xp.int32);base=xp.empty_like(tids);packed=xp.empty_like(tids)
    values=xp.empty(records,dtype=xp.int16);error=xp.zeros(1,dtype=xp.int32)
    module.get_function('csv_parse')(grid,(128,),(buf,starts,colons,line_ends,offsets,np.int64(transactions),
        np.int32(membership_scale),tids,base,packed,values,error))
    parse_error=int(error.get()[0])
    if parse_error:
        raise ValueError(f'GPU CSV parse error {parse_error}: numeric item.region terms and exact memberships are required')
    if grouped:del buf,ends,colons,line,starts,line_ends,counts,offsets
    if not records:
        return {},{'levels':[0],'candidates':0,'list_matches':0,'transactions':transactions*repeat,'base_items':0,'input_bytes':input_bytes}
    label_codes=xp.unique(packed);labels=[unpack_region(int(v)) for v in label_codes.get()]
    region=xp.searchsorted(label_codes,packed).astype(xp.int32);R=len(labels)
    max_base=int(base.max().get())+1
    if max_base*R>20_000_000:
        raise ValueError('GPU CSV item IDs are too sparse; use the dictionary API for sparse/string labels')
    codes=base.astype(xp.int64)*R+region
    del base,region,packed,label_codes
    upper=record_supports_gpu(xp,codes,values,max_base*R,membership_scale)
    base_items=int(xp.count_nonzero(upper.reshape(max_base,R).sum(axis=1)).get())
    keep=xp.ones(len(codes),dtype=xp.bool_) if duplicate_policy=='error' else (upper[codes]*repeat>=threshold)&(values>0)
    codes,tids,values=codes[keep],tids[keep],values[keep]
    del upper,keep
    if not len(codes):
        return {},{'levels':[0,0],'candidates':0,'list_matches':0,'transactions':transactions*repeat,'base_items':base_items,'input_bytes':input_bytes}
    keys=(codes.astype(xp.uint64)<<xp.uint64(32))|tids.astype(xp.uint64)
    order=xp.argsort(keys);keys,values=keys[order],values[order]
    del codes,tids,order
    first=xp.empty(len(keys),dtype=xp.bool_);first[0]=True;first[1:]=keys[1:]!=keys[:-1]
    distinct=xp.flatnonzero(first)
    if len(distinct)!=len(keys):
        dedup=xp.empty(len(distinct),dtype=xp.int16);error.fill(0)
        module.get_function('dedup_memberships')(((len(distinct)+127)//128,),(128,),
            (values,distinct,np.int64(len(distinct)),np.int64(len(keys)),np.int32(duplicate_policy=='error'),dedup,error))
        if int(error.get()[0]):raise ValueError('Conflicting duplicate term memberships')
        values=dedup;keys=keys[distinct]
    codes=(keys>>xp.uint64(32)).astype(xp.int64)
    tids=(keys&xp.uint64(0xffffffff)).astype(xp.int32)
    del keys,first,distinct
    exact=record_supports_gpu(xp,codes,values,max_base*R,membership_scale)
    frequent=xp.flatnonzero(exact*repeat>=threshold)
    host_codes=frequent.get();host_supports=(exact[frequent]*repeat).get()
    if not len(host_codes):
        return {},{'levels':[0,0],'candidates':0,'list_matches':0,'transactions':transactions*repeat,'base_items':base_items,'input_bytes':input_bytes}
    term_map=xp.full(max_base*R,-1,dtype=xp.int32);term_map[frequent]=xp.arange(len(frequent),dtype=xp.int32)
    mapped=term_map[codes];keep=(mapped>=0)&(values>0)
    record_term=mapped[keep];source_tids=tids[keep];source_values=values[keep]
    del mapped,keep,codes,tids,values,exact,frequent,term_map
    source_counts=xp.bincount(record_term,minlength=len(host_codes)).get().astype(np.int64)
    terms=[(str(int(code)//R),labels[int(code)%R]) for code in host_codes]
    by_item=defaultdict(list)
    for term in terms:by_item[term[0]].append(term)
    bases=list(by_item)
    single_t,single_v,single_offsets,_=repeat_gpu_lists(xp,source_tids,source_values,source_counts,
        np.ones(len(terms),dtype=np.int64),repeat,transactions)
    singles={term:(single_t[single_offsets[i]:single_offsets[i+1]],single_v[single_offsets[i]:single_offsets[i+1],None])
             for i,term in enumerate(terms)}
    initial={(term,):int(support)/membership_scale for term,support in zip(terms,host_supports)}
    groups={b:singles[by_item[b][0]] for b in bases if len(by_item[b])==1} if grouped else {}
    if grouped and any(len(by_item[b])>1 for b in bases):
        base_ids={b:i for i,b in enumerate(bases)}
        term_base=np.array([base_ids[t[0]] for t in terms],dtype=np.int32)
        term_region=np.array([by_item[t[0]].index(t) for t in terms],dtype=np.int32)
        widths=np.array([len(by_item[b]) for b in bases],dtype=np.int64)
        db=xp.asarray(term_base);dc=xp.asarray(term_region)
        select_multi=xp.asarray(widths>1)[db[record_term]]
        selected_terms=record_term[select_multi];selected_tids=source_tids[select_multi];selected_values=source_values[select_multi]
        group_base=db[selected_terms]
        group_keys=(group_base.astype(xp.uint64)<<xp.uint64(32))|selected_tids.astype(xp.uint64)
        order=xp.argsort(group_keys)
        group_keys,group_terms,group_values=group_keys[order],selected_terms[order],selected_values[order]
        first=xp.empty(len(group_keys),dtype=xp.bool_);first[0]=True;first[1:]=group_keys[1:]!=group_keys[:-1]
        row=xp.cumsum(first,dtype=xp.int32)-1
        unique=group_keys[first];gb=(unique>>xp.uint64(32)).astype(xp.int32)
        grouped_counts=xp.bincount(gb,minlength=len(bases)).get().astype(np.int64)
        goff=np.r_[np.int64(0),np.cumsum(grouped_counts,dtype=np.int64)]
        voff=np.r_[np.int64(0),np.cumsum(grouped_counts*widths,dtype=np.int64)]
        rowbase=(group_keys>>xp.uint64(32)).astype(xp.int32)
        at=xp.asarray(voff[:-1])[rowbase]+(row-xp.asarray(goff[:-1])[rowbase])*xp.asarray(widths)[rowbase]+dc[group_terms]
        gv=xp.zeros(int(voff[-1]),dtype=xp.int16);gv[at]=group_values
        gt=(unique&xp.uint64(0xffffffff)).astype(xp.int32)
        gt,gv,goff,voff=repeat_gpu_lists(xp,gt,gv,grouped_counts,widths,repeat,transactions)
        groups.update({b:(gt[goff[i]:goff[i+1]],gv[voff[i]:voff[i+1]].reshape(int(grouped_counts[i]*repeat),int(widths[i])))
                for i,b in enumerate(bases) if widths[i]>1})
        del group_keys,group_terms,group_values,order,first,row,unique,gb,rowbase,at,db,dc,select_multi,selected_terms,selected_tids,selected_values,group_base
    del record_term,source_tids,source_values
    result,stats=pooled_levels(xp,threshold,bases,by_item,singles,groups,initial,membership_scale,transactions*repeat)
    stats.update(transactions=transactions*repeat,base_items=base_items,input_bytes=input_bytes,
                 timing_scope='full file read + GPU parse/dedup + all input layouts + mining + result download',
                 input_format='original fuzzy CSV',repeat=repeat)
    return result,stats


def measure_file(path,threshold,xp,grouped,runs,repeat=1,membership_scale=10,fmt='fuzzy',duplicate_policy='max',allocator='fresh'):
    times=[]
    for run in range(runs+1):
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
            if allocator=='fresh':
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        start=time.perf_counter()
        if xp is not np and fmt=='fuzzy':
            results,stats=mine_file_gpu(path,threshold,xp,grouped,repeat,membership_scale,duplicate_policy)
        else:
            database,transactions=read_database(path,fmt,repeat,duplicate_policy)
            prepared=prepare_database(database,membership_scale)
            results,stats=mine(database,threshold,xp,grouped,membership_scale,prepared)
            stats.update(transactions=transactions,base_items=len({t[0] for t in database}),timing_scope='full file read + parse/packing + mining + result download')
        if xp is not np:xp.cuda.Stream.null.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        stats['allocation_policy']=allocator
        if run:times.append(elapsed)
    return results,stats,times


def prepare_database(database, membership_scale=10):

    if not 1 <= membership_scale <= 32767:
        raise ValueError('membership_scale must be between 1 and 32767')
    terms, host_tids, host_values, lengths = [], [], [], []
    for term in sorted(database):
        tids, values = database[term]
        tids, values = np.asarray(tids), np.asarray(values)
        if tids.ndim != 1 or values.shape != tids.shape:
            raise ValueError('Occurrence lists must be matching one-dimensional arrays')
        if not np.issubdtype(tids.dtype, np.integer) or np.any(tids < 0) or np.any(np.diff(tids) <= 0):
            raise ValueError('Transaction IDs must be unique sorted nonnegative integers')
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError('Memberships must be finite and in [0,1]')
        if len(tids) and int(tids[-1]) > 2147483647:
            raise ValueError('Transaction IDs exceed the int32 representation')
        ticks = np.rint(values * membership_scale)
        if not np.allclose(values * membership_scale, ticks, rtol=0, atol=1e-8):
            raise ValueError('Membership precision exceeds --membership-scale')
        positive = ticks > 0
        if not np.any(positive):
            continue
        terms.append(term)
        host_tids.append(tids[positive].astype(np.int64))
        host_values.append(ticks[positive].astype(np.int64))
        lengths.append(int(positive.sum()))
    offsets = np.cumsum([0] + lengths, dtype=np.int64)
    flat_tids = np.concatenate(host_tids).astype(np.int32) if terms else np.empty(0, dtype=np.int32)
    flat_values = np.concatenate(host_values).astype(np.int16) if terms else np.empty(0, dtype=np.int16)


    by_base = defaultdict(list)
    for index, term in enumerate(terms):
        by_base[term[0]].append(index)
    raw_groups = {}
    for base, indices in by_base.items():
        if len(indices) == 1:
            index = indices[0]
            raw_groups[base] = (host_tids[index], host_values[index][:, None], {terms[index]: 0})
            continue
        tids = np.unique(np.concatenate([host_tids[index] for index in indices]))
        values = np.zeros((len(tids), len(indices)), dtype=np.int16)
        columns = {}
        for col, index in enumerate(indices):
            values[np.searchsorted(tids, host_tids[index]), col] = host_values[index]
            columns[terms[index]] = col
        raw_groups[base] = tids, values, columns
    raw_bases = sorted(raw_groups)
    raw_offsets_t = np.r_[np.int64(0), np.cumsum([len(raw_groups[b][0]) for b in raw_bases], dtype=np.int64)]
    raw_offsets_v = np.r_[np.int64(0), np.cumsum([raw_groups[b][1].size for b in raw_bases], dtype=np.int64)]
    raw_tids = np.concatenate([raw_groups[b][0] for b in raw_bases]).astype(np.int32) if raw_bases else np.empty(0, dtype=np.int32)
    raw_values = np.concatenate([raw_groups[b][1].ravel() for b in raw_bases]).astype(np.int16) if raw_bases else np.empty(0, dtype=np.int16)
    raw_layout = (raw_tids, raw_values, raw_offsets_t, raw_offsets_v,
                  np.array([raw_groups[b][1].shape[1] for b in raw_bases], dtype=np.int64),
                  {base: i for i, base in enumerate(raw_bases)})
    return terms, host_tids, host_values, offsets, flat_tids, flat_values, raw_groups, raw_layout


def mine(database, min_sup, xp=np, grouped=True, membership_scale=10, prepared=None, pooled=True):

    from decimal import Decimal, ROUND_CEILING
    if not math.isfinite(min_sup) or min_sup <= 0:
        raise ValueError('minSup must be finite and positive')
    if not isinstance(membership_scale, int) or membership_scale < 1:
        raise ValueError('membership_scale must be a positive integer')
    threshold = int((Decimal(str(min_sup)) * membership_scale).to_integral_value(rounding=ROUND_CEILING))
    terms, host_tids, host_values, offsets, flat_tids, flat_values, raw_groups, raw_layout = (
        prepare_database(database, membership_scale) if prepared is None else prepared)
    if not terms:
        return {}, {'levels': [0], 'candidates': 0, 'list_matches': 0}

    dt = xp.asarray(flat_tids, dtype=xp.int32)
    dv = xp.asarray(flat_values, dtype=xp.int16)
    if xp is np:
        supports = np.add.reduceat(dv, offsets[:-1])
    else:
        kernel = xp.RawModule(code=CUDA_SOURCE).get_function('singletons')
        supports = xp.empty(len(terms), dtype=xp.int64)
        kernel((len(terms),), (128,), (dv, xp.asarray(offsets), supports))
        supports = supports.get()


    flat_lists = None
    if xp is not np and not grouped:
        lengths = np.diff(offsets)
        record_terms = xp.repeat(xp.arange(len(terms), dtype=xp.int32), lengths.tolist())
        selected_records = xp.asarray(supports >= threshold)[record_terms]
        keys = (record_terms[selected_records].astype(xp.uint64) << xp.uint64(32)) | dt[selected_records].astype(xp.uint64)
        order = xp.argsort(keys)
        flat_t = dt[selected_records][order]
        flat_v = dv[selected_records][order]
        surviving = np.flatnonzero(supports >= threshold)
        flat_offsets = np.r_[np.int64(0), np.cumsum(lengths[surviving], dtype=np.int64)]
        flat_lists = {terms[index]: (flat_t[flat_offsets[k]:flat_offsets[k+1]],
                                    flat_v[flat_offsets[k]:flat_offsets[k+1], None])
                      for k,index in enumerate(surviving)}

    results, singles, by_item, host_singles = {}, {}, defaultdict(list), {}
    for index, term in enumerate(terms):
        support = int(supports[index])
        if support >= threshold:
            start, end = offsets[index:index+2]
            singles[term] = flat_lists[term] if flat_lists is not None else (dt[start:end], dv[start:end, None])
            host_singles[term] = (host_tids[index], host_values[index])
            by_item[term[0]].append(term)
            results[(term,)] = support / membership_scale
    bases = sorted(by_item)
    groups = {}
    if grouped and xp is not np:
        groups = build_gpu_groups(xp, bases, by_item, singles, raw_groups, raw_layout)
    elif grouped:


        grouped_tids, grouped_values, shapes, grouped_bases = [], [], [], []
        for base in bases:
            ts = by_item[base]
            if len(ts) == 1:
                groups[base] = singles[ts[0]]
                continue
            grouped_bases.append(base)
            raw_tids, raw_values, raw_columns = raw_groups[base]
            values = raw_values[:, [raw_columns[t] for t in ts]]
            positive = np.any(values > 0, axis=1)
            tids, values = raw_tids[positive], values[positive]
            grouped_tids.append(tids)
            grouped_values.append(values.ravel())
            shapes.append((len(tids), len(ts)))
        if shapes:
            all_tids = xp.asarray(np.concatenate(grouped_tids), dtype=xp.int32)
            all_values = xp.asarray(np.concatenate(grouped_values), dtype=xp.int16)
            ti = vi = 0
            for base, (n, width) in zip(grouped_bases, shapes):
                groups[base] = (all_tids[ti:ti+n], all_values[vi:vi+n*width].reshape(n, width))
                ti, vi = ti+n, vi+n*width
    if xp is not np and pooled:
        output, stats = pooled_levels(xp, threshold, bases, by_item, singles, groups, results, membership_scale, int(flat_tids.max())+1)
        stats['singleton_layout'] = 'explicit per-term views + GPU key sort' if not grouped else 'GPU base/region projection'
        return output, stats
    matcher = Matcher(xp, threshold, results, membership_scale)


    def jobs(left, right, choices):
        for start in range(0, len(choices), 32):
            yield left, right, choices[start:start+32]
    def pair_jobs():
        for i, j in itertools.combinations(bases, 2):
            if grouped:
                choices = [((a, b), ai, bi) for ai, a in enumerate(by_item[i])
                           for bi, b in enumerate(by_item[j])]
                yield from jobs(groups[i], groups[j], choices)
            else:
                for a in by_item[i]:
                    for b in by_item[j]:
                        yield singles[a], singles[b], [((a, b), 0, 0)]


    current = (matcher.run(pair_jobs()) if xp is np
               else matcher.pairs(bases, by_item, groups, singles, grouped))
    pairs = set(current)
    neighbors = defaultdict(set)
    for a, b in pairs:
        neighbors[a].add(b)
    if xp is not np and current:
        surviving_terms = sorted(singles)
        term_ids = {term: index for index, term in enumerate(surviving_terms)}
        base_ids = {base: index for index, base in enumerate(bases)}
        term_bases = np.array([base_ids[t[0]] for t in surviving_terms], dtype=np.int32)
        term_columns = np.array([by_item[t[0]].index(t) for t in surviving_terms], dtype=np.int32)
        pair_matrix = np.zeros((len(surviving_terms), len(surviving_terms)), dtype=np.bool_)
        adjacency = [[] for _ in surviving_terms]
        for a, b in pairs:
            ai, bi = term_ids[a], term_ids[b]
            pair_matrix[ai, bi] = True
            adjacency[ai].append(bi)
        neighbor_offsets = np.cumsum([0] + [len(row) for row in adjacency], dtype=np.int64)
        neighbor_ids = np.array([t for row in adjacency for t in sorted(row)], dtype=np.int32)
    levels = [len(singles), len(current)]

    while current:
        def extension_jobs():
            for prefix, occurrence in current.items():


                allowed = set(neighbors[prefix[0]])
                for term in prefix[1:]:
                    allowed.intersection_update(neighbors[term])
                extensions = defaultdict(list)
                for term in sorted(allowed):
                    extensions[term[0]].append(term)
                for base, terms in extensions.items():
                    if grouped:
                        choices = [(prefix + (term,), 0, by_item[base].index(term)) for term in terms]
                        yield from jobs(occurrence, groups[base], choices)
                    else:
                        for term in terms:
                            yield occurrence, singles[term], [(prefix + (term,), 0, 0)]
        if xp is np:
            current = matcher.run(extension_jobs())
        else:
            current = matcher.extensions(current, surviving_terms, term_ids, term_bases,
                                         term_columns, pair_matrix, neighbor_offsets,
                                         neighbor_ids, bases, groups, singles, grouped)
        if current:
            levels.append(len(current))

    return results, {'levels': levels, 'candidates': matcher.candidates,
                     'list_matches': matcher.jobs}


REFERENCE = {
    'retail': ('retail', 1, 88162, [200, 300, 500, 700, 1000],
               [541.30, 336.95, 212.05, 166.80, 155.90],
               [247.17, 159.69, 107.09, 91.65, 65.78]),
    'kosarak': ('kosarak', 1, 990002, [4000, 4500, 5000, 6000, 10000],
                [707.50, 598.75, 566.70, 498.20, 390.70],
                [300.64, 279.79, 292.11, 248.35, 157.86]),
    'T10I4D1M': ('T10I4D100K', 10, 1000000, [10000, 20000, 30000, 40000, 50000],
                 [240.00, 175.50, 145.20, 132.05, 129.40],
                 [131.15, 76.63, 81.57, 60.30, 55.99]),
    'pumsb': ('pumsb', 1, 49046, [100000, 110000, 120000, 130000, 200000],
              [3086.70, 1907.75, 1208.65, 826.90, 239.20],
              [1696.54, 1053.09, 688.83, 472.51, 102.00]),
}


def measure(database, threshold, xp, grouped, runs, membership_scale, prepared=None):
    if prepared is None:
        prepared = prepare_database(database, membership_scale)
    times = []
    for run in range(runs + 1):
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        results, stats = mine(database, threshold, xp, grouped, membership_scale, prepared)
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if run:
            times.append(elapsed)
    return results, stats, times


def benchmark(args, xp):

    selected = list(REFERENCE) if args.dataset == 'all' else [args.dataset]
    destination = args.benchmark_output
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with destination.open('w') as output:
        for name in selected:
            source, repeat, expected_n, thresholds, naive_ref, grouped_ref = REFERENCE[name]
            input_path = args.data_dir / f'Fuzzy_{source}.csv'
            for point, threshold in enumerate(thresholds):
                expected = None
                candidate_count = None

                for grouped in ([False, True] if point % 2 == 0 else [True, False]):
                    results, stats, times = measure_file(input_path, threshold / 10, xp, grouped,
                        args.runs, repeat, args.membership_scale, args.format, args.duplicate_policy, args.allocator)
                    transactions = stats['transactions']
                    if transactions != expected_n:
                        raise ValueError(f'{name}: expected {expected_n} transactions, found {transactions}')
                    if expected is None:
                        expected, candidate_count = results, stats['candidates']
                    elif results != expected or stats['candidates'] != candidate_count:
                        raise AssertionError(f'{name} at {threshold}: grouped and naive disagree')
                    reference = (grouped_ref if grouped else naive_ref)[point]
                    record = {'dataset': name, 'mode': 'grouped' if grouped else 'naive',
                              'backend': args.backend, 'device': args.device, 'candidate_backend': candidate_backend() if args.backend == 'cuda' else 'python',
                              'transactions': transactions, 'base_items': stats['base_items'],
                              'minSup_x10': threshold, 'patterns': len(results),
                              'median_ms': statistics.median(times), 'runs_ms': times,
                              'reference_ms': reference, 'relative_error': statistics.median(times)/reference-1,
                              'within_20_percent': abs(statistics.median(times)/reference-1)<=.2, 'discarded_warmups': 1,
                              'membership_scale': args.membership_scale, 'duplicate_policy': args.duplicate_policy,
                              'timing_scope': stats['timing_scope'],
                              **stats}
                    records.append(record)
                    line = json.dumps(record)
                    output.write(line + '\n')
                    output.flush()
                    print(line, flush=True)
    write_benchmark_reports(records, destination)


def write_benchmark_reports(records, destination):

    destination = Path(destination)
    selected = list(dict.fromkeys(r['dataset'] for r in records))
    fields = ['dataset', 'mode', 'minSup_x10', 'median_ms', 'reference_ms',
              'patterns', 'candidates', 'list_matches', 'transactions', 'base_items',
              'relative_error', 'within_20_percent', 'discarded_warmups', 'timing_scope']
    with destination.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    comparisons = []
    for name in selected:
        for mode in ('naive', 'grouped'):
            rows = sorted((r for r in records if r['dataset'] == name and r['mode'] == mode),
                          key=lambda r: r['minSup_x10'])
            measured = np.array([r['median_ms'] for r in rows])
            reference = np.array([r['reference_ms'] for r in rows])
            if len(rows) < 2:
                continue
            comparisons.append({
                'dataset': name, 'mode': mode,
                'minSup_x10': [r['minSup_x10'] for r in rows],
                'measured_ms': measured.tolist(), 'reference_ms': reference.tolist(),
                'shape_correlation': float(np.corrcoef(measured, reference)[0, 1]),
                'normalized_rmse': float(np.sqrt(np.mean((measured/measured[0] - reference/reference[0])**2))),
                'matching_adjacent_directions': int(np.count_nonzero(np.sign(np.diff(measured)) == np.sign(np.diff(reference)))),
                'adjacent_steps': len(rows) - 1,
                'note': 'Comparison metrics only; measured runtimes are unchanged.',
            })
    destination.with_suffix('.comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')


    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for normalized in (False, True):
        fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), squeeze=False)
        for ax, name in zip(axes[0], selected):
            for mode, color in [('naive', 'tab:orange'), ('grouped', 'tab:blue')]:
                rows = sorted((r for r in records if r['dataset'] == name and r['mode'] == mode),
                              key=lambda r: r['minSup_x10'])
                measured = np.array([r['median_ms'] for r in rows])
                reference = np.array([r['reference_ms'] for r in rows])
                if normalized:
                    measured, reference = measured / measured[0], reference / reference[0]
                x = [r['minSup_x10'] for r in rows]
                ax.plot(x, measured, 'o-', color=color, label=f'{mode}: measured')
                ax.plot(x, reference, '--', color=color, alpha=.6, label=f'{mode}: reference')
            ax.set_title(name)
            ax.set_xlabel('minSup ×10')
            ax.set_ylabel('Runtime / first threshold' if normalized else 'Runtime (ms)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        suffix = '.normalized.png' if normalized else '.png'
        fig.savefig(destination.with_suffix(suffix), dpi=160)
        plt.close(fig)


def self_test(xp):

    import tempfile
    rng = np.random.default_rng(2026)
    cases = 0
    for trial in range(16):
        transactions = 513 if trial >= 12 else 9
        dense = rng.integers(0, 11, size=(transactions, 4, 3))
        threshold = [13, 20, 31, 90][trial % 4] * (30 if transactions > 9 else 1)
        database = {}
        for item in range(4):
            for region in range(3):
                values = dense[:, item, region]
                tids = np.flatnonzero(values)
                database[(str(item), str(region))] = (tids, values[tids] / 10)
        oracle = {}
        for length in range(1, 5):
            for items in itertools.combinations(range(4), length):
                for regions in itertools.product(range(3), repeat=length):
                    terms = tuple((str(i), str(r)) for i, r in zip(items, regions))
                    support = int(np.minimum.reduce([dense[:, i, r] for i, r in zip(items, regions)]).sum())
                    if support >= threshold:
                        oracle[terms] = support / 10
        for backend in ([np] if xp is np else [np, xp]):
            stats = []
            for grouped in (False, True):
                actual, detail = mine(database, threshold / 10, backend, grouped)
                if actual != oracle:
                    raise AssertionError(f'Oracle disagreement: trial={trial}, backend={backend.__name__}, grouped={grouped}')
                stats.append(detail)
                cases += 1
            assert stats[0]['candidates'] == stats[1]['candidates']
            assert stats[1]['list_matches'] <= stats[0]['list_matches']
    if xp is not np:


        global NAIVE_KERNEL
        previous_kernel = NAIVE_KERNEL
        try:
            for kernel_name in ('serial', 'atomic', 'csr_serial', 'csr_parallel'):
                NAIVE_KERNEL = kernel_name
                for shift, scale in ((0, 10), (1 << 28, 20)):
                    shifted = {term: (tids + shift, values) for term, (tids, values) in database.items()}
                    actual, _ = mine(shifted, threshold / 10, xp, False, membership_scale=scale)
                    assert actual == oracle, (kernel_name, shift, scale)
                    cases += 1
        finally:
            NAIVE_KERNEL = previous_kernel
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'fixture.csv'
        source.write_text('a.L a.L b.H:0.2 0.7 0.5\na.L b.H:1 0.3\n')
        database, n = read_database(source, repeat=3)
        assert n == 6
        for grouped in (False, True):
            actual, _ = mine(database, 2.4, xp, grouped)
            assert actual[(("a", "L"), ("b", "H"))] == 2.4
            assert actual[(("a", "L"),)] == 5.1
            assert mine(database, 2.41, xp, grouped)[0].get((("a", "L"), ("b", "H"))) is None
        try:
            read_database(source, duplicate_policy='error')
        except ValueError:
            pass
        else:
            raise AssertionError('Conflicting duplicates were accepted in strict mode')
    if xp is not np:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'numeric.csv'
            source.write_text('2.L 2.L 10.H:0.2 0.7 0.5\n\n2.L 10.H:1 0.3')
            database, _ = read_database(source, repeat=3)
            expected, _ = mine(database, 0.8, np)
            for grouped in (False, True):
                actual, _ = mine_file_gpu(source, 0.8, xp, grouped, repeat=3)
                assert actual == expected, ('GPU CSV/repeat', grouped)
                cases += 1
                try:
                    mine_file_gpu(source, 100, xp, grouped, duplicate_policy='error')
                except ValueError:
                    pass
                else:
                    raise AssertionError('GPU parser accepted conflicting duplicates')
    if xp is not np:
        tids = np.arange(70000, dtype=np.int32)
        wide_database = {(str(i), 'H'): (tids, np.ones(len(tids))) for i in range(3)}
        for grouped in (False, True):
            actual, _ = mine(wide_database, 69999, xp, grouped, membership_scale=32767)
            assert len(actual) == 7 and all(value == 70000 for _, value in actual.items())
            cases += 1
        regions_database = {(str(i), str(r)): (np.arange(40), np.full(40, (r+1)/5))
                            for i in range(3) for r in range(5)}
        expected, _ = mine(regions_database, 4, np)
        actual, _ = mine(regions_database, 4, xp, True)
        assert actual == expected
        cases += 1
    assert mine({}, 1, xp)[0] == {}
    print(json.dumps({'self_test': 'passed', 'oracle_comparisons': cases,
                      'backend': 'cpu' if xp is np else 'cpu+cuda', 'candidate_backend': 'python' if xp is np else candidate_backend(),
                      'edge_checks': ['empty database', 'duplicate policy', 'repeat', 'threshold ceiling']}))


def main(grouped=True):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('database', nargs='?')
    p.add_argument('minSup', nargs='?', type=float)
    p.add_argument('--format', choices=['fuzzy', 'triples'], default='fuzzy')
    p.add_argument('--duplicate-policy', choices=['max', 'error'], default='max', help='fuzzy-set union or reject conflicting repeated terms')
    p.add_argument('--support-scale', type=float, default=10, help='threshold divisor; default 10 matches manuscript')
    p.add_argument('--membership-scale', type=int, default=10, help='exact fixed-point membership denominator')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--allocator', choices=['fresh','reuse'], default='fresh', help='fresh includes standalone GPU allocation costs')
    p.add_argument('--backend', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--naive-kernel',choices=['auto','serial','atomic','csr_serial','csr_parallel'],default='auto',help='GPU mapping for the flat baseline')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--check-cuda', action='store_true')
    p.add_argument('--self-test', action='store_true', help='compare both miners with exhaustive enumeration')
    p.add_argument('--output', type=Path)
    p.add_argument('--benchmark', action='store_true', help='run both miners over the supplied reference thresholds')
    p.add_argument('--dataset', choices=['all'] + list(REFERENCE), default='all')
    p.add_argument('--data-dir', type=Path, default=Path('datasets'))
    p.add_argument('--benchmark-output', type=Path, default=Path('results/benchmark.jsonl'))
    args = p.parse_args()
    global NAIVE_KERNEL
    NAIVE_KERNEL=args.naive_kernel
    xp = np
    if args.backend == 'cuda' or args.check_cuda:
        try:
            import cupy as xp
            xp.cuda.Device(args.device).use()
            probe = xp.arange(1024, dtype=xp.float64)
            assert scalar(probe.sum()) == 523776
            xp.cuda.Stream.null.synchronize()
            print(json.dumps({'cuda': 'ok', 'device': args.device,
                              'runtime': xp.cuda.runtime.runtimeGetVersion()}))
        except Exception as exc:
            p.exit(1, f'CUDA initialization failed: {exc}\n')
    if args.check_cuda:
        return
    if args.repeat < 1 or args.runs < 1 or args.membership_scale < 1 or not math.isfinite(args.support_scale) or args.support_scale <= 0:
        p.error('repeat, runs, membership-scale and support-scale must be positive')
    if args.self_test:
        self_test(xp)
        return
    if args.benchmark:
        benchmark(args, xp)
        return
    if args.database is None or args.minSup is None:
        p.error('database and minSup are required')
    results, stats, times = measure_file(args.database, args.minSup / args.support_scale,
        xp, grouped, args.runs, args.repeat, args.membership_scale, args.format, args.duplicate_policy, args.allocator)
    transactions = stats['transactions']
    print(json.dumps({'mode': 'grouped' if grouped else 'naive', 'backend': args.backend,
                      'transactions': transactions, 'minSup': args.minSup,
                      'candidate_backend': candidate_backend() if args.backend == 'cuda' else 'python', 'support_scale': args.support_scale, 'membership_scale': args.membership_scale, 'patterns': len(results),
                      'median_ms': statistics.median(times), 'runs_ms': times,
                      'discarded_warmups': 1, **stats}))
    if args.output:
        with args.output.open('w') as f:
            for pattern, support in sorted(results.items()):
                f.write(json.dumps({'pattern': pattern, 'support': support}) + '\n')


if __name__ == '__main__':
    main()
