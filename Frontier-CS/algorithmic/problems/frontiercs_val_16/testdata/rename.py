import os

# 1. Delete files 1-18 (in/ans)
for i in range(1, 19):
    for ext in ["in", "ans"]:
        fname = f"{i}.{ext}"
        if os.path.exists(fname):
            os.remove(fname)
            print(f"Deleted {fname}")

# 2. Renumber files 19-28 to 1-10
new_index = 1
for i in range(19, 29):
    old_in = f"{i}.in"
    old_ans = f"{i}.ans"
    new_in = f"{new_index}.in"
    new_ans = f"{new_index}.ans"

    if os.path.exists(old_in):
        os.rename(old_in, new_in)
        print(f"{old_in} -> {new_in}")
    if os.path.exists(old_ans):
        os.rename(old_ans, new_ans)
        print(f"{old_ans} -> {new_ans}")

    new_index += 1
