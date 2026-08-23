from datasets import load_dataset


print("=" * 50)
print("LOADING AGENTLANS READABILITY DATASET")
print("=" * 50)

agentlans = load_dataset("agentlans/readability")

print(agentlans)
print("\nFirst example:")
print(agentlans["train"][0])


print("\n" + "=" * 50)
print("LOADING README++ DATASET")
print("=" * 50)

readme = load_dataset("UniversalCEFR/readme_en")

print(readme)
print("\nFirst example:")
print(readme["train"][0])