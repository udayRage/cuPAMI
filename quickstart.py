import cupami as cuPAMI


data = cuPAMI.TransactionalDatabase(
    transactions=10000,
    items=50,
    min_length=2,
    max_length=10,
).create()

miner = cuPAMI.cuApriori(data, minSup=0.01)
miner.mine()
patterns = miner.getPatterns()

print("Transactions:", len(data))
print("Patterns:", len(patterns))
print(f"Mining time: {miner.getRuntime():.4f} seconds")
print(f"Host RAM: {miner.getMemoryRSS() / 1024**2:.1f} MiB")

print("\nSample patterns and support counts:")
for pattern, support in list(patterns.items())[:5]:
    print(pattern.replace("\t", ", "), ":", support)
