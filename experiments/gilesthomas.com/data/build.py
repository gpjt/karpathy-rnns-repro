from pathlib import Path

SOURCE_PATH = Path("/home/giles/Dev/www.gilesthomas.com/sources/posts")
OUTPUT_PATH = Path(__file__).resolve().parent / "input.txt"

OUTPUT_PATH.write_text("")

for filename in sorted(SOURCE_PATH.iterdir()):
    path = SOURCE_PATH / filename
    data = path.read_text()
    header = data.split("---")[1]
    if not "state: published" in header:
        continue
    with open(OUTPUT_PATH, "a") as f:
        f.write("<|article-start|>\n\n")
        f.write(data)
        f.write("\n\n<|article-end|>\n\n")
