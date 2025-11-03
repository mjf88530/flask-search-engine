import re, csv, pickle
from array import array
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Paths (robust to working directory) -------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "large-sample"          # folder with *.txt
INDEX_PATH = BASE_DIR / "index.dat"
TERMS_PATH = BASE_DIR / "terms.csv"
DOC_PATH = BASE_DIR / "document.csv"
HISTORY_PATH = BASE_DIR / "history.pkl"


def regex(mystr: str) -> str:
    """Return a space-joined, lowercased alphanumeric-only string."""
    matches = re.findall(r"\w+", mystr.lower())
    return " ".join(matches)


def main():
    """
    Create/refresh the data artifacts (index.dat, document.csv, terms.csv) if
    any are missing. Builds a TF-IDF matrix over all *.txt in DATABASE_PATH.
    """
    p = DATABASE_PATH

    file_stat = []
    corpus = []

    # Only (re)build if any artifact is missing.
    if not INDEX_PATH.exists() or not TERMS_PATH.exists() or not DOC_PATH.exists():
        # Validate data folder
        if not p.exists():
            raise FileNotFoundError(
                f"Data folder not found: {p}. Make sure 'large-sample/' sits next to searcher.py."
            )

        files = list(p.glob("*.txt"))
        if not files:
            raise FileNotFoundError(
                f"No .txt files found in {p}. Add documents before running."
            )

        # Load and clean documents
        for file in files:
            file_stat.append((file.name, file.stat().st_size))
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            cleaned = regex(content).strip()
            if cleaned:
                corpus.append(cleaned)

        if not corpus:
            raise ValueError(
                "All documents became empty after cleaning. Check preprocessing or source files."
            )

        print("Start vectorizing...")

        # Create tf-idf matrix 
        vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r"(?u)\b\w+\b",
            min_df=0.01
        )

        X = vectorizer.fit_transform(corpus)
        loc_dict = vectorizer.vocabulary_

        print("Finish vectorizing.")

        create_index(X)
        create_terms(loc_dict)
        create_doc(file_stat)
    else:
        print("All data files exist.")


def create_terms(word_dict):
    """Store every word and its column index to terms.csv.  ROW: term, term_column"""
    if not TERMS_PATH.exists():
        print("terms.csv creating...")
        with open(TERMS_PATH, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for term, col in word_dict.items():
                writer.writerow([term, col])
        print("terms.csv created.")
    else:
        print("terms.csv already exists")


def create_doc(file_stat):
    """Store file stats (name, size) to document.csv.  ROW: document_name, size"""
    if not DOC_PATH.exists():
        print("document.csv creating...")
        with open(DOC_PATH, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for name, size in file_stat:
                writer.writerow([name, size])
        print("document.csv created.")
    else:
        print("document.csv exists")


def create_index(matrix):
    """
    Create index.dat storing nonzero tf-idf entries.

    INDEX FORMAT (double array):
        [DOC_ID, TERM_COL, TF_IDF, DOC_ID, TERM_COL, TF_IDF, ...]
    """
    if not INDEX_PATH.exists():
        print("Writing index.dat...")
        with open(INDEX_PATH, "wb") as f:
            for i, j in zip(*matrix.nonzero()):
                binary_array = array("d", [i, j, float(matrix[i, j])])
                binary_array.tofile(f)
        print("index.dat created.")
    else:
        print("index.dat exists")


def id_to_name(doc_id: int) -> str:
    """Convert document id to its file name."""
    file_name = []
    with open(DOC_PATH, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_name.append(row)
    return file_name[int(doc_id)][0]


def query(text: str):
    """
    Search the index for a single token `text`.
    Return list of tuples (doc_name, term, tf_idf) sorted by tf_idf desc.
    """
    doc = []
    exist = False
    column = None

    # look up term column
    with open(TERMS_PATH, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for term, col in reader:
            if text == term:
                column = col
                exist = True
                break
    if not exist:
        return None

    print("start searching...")

    # use history if available
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        if text in history:
            print("history found.")
            for doc_name, tf_idf in history[text]:
                doc.append((doc_name, text, tf_idf))
            return doc

    # scan index.dat
    with open(INDEX_PATH, "rb") as f:
        bin_arr = array("d")
        bin_arr.frombytes(f.read())
        for i in range(0, len(bin_arr), 3):
            if int(column) == int(bin_arr[i + 1]):
                doc.append((id_to_name(int(bin_arr[i])), text, bin_arr[i + 2]))

    doc.sort(key=lambda x: x[2], reverse=True)
    store_history(HISTORY_PATH, doc)
    return doc


def store_history(p: Path, doc):
    """Write/update history.pkl with {word: [(doc_name, tf_idf), ...]}."""
    history_dict = {}

    if not p.exists():
        with open(p, "wb") as f:
            for name, text, tf_idf in doc:
                history_dict = history(name, text, tf_idf, history_dict)
            pickle.dump(history_dict, f)
        print("history created")
    else:
        with open(p, "rb") as f:
            history_dict = pickle.load(f)
        with open(p, "wb") as f:
            for name, text, tf_idf in doc:
                history_dict = history(name, text, tf_idf, history_dict)
            pickle.dump(history_dict, f)


def history(doc_name, text, tf_idf, d: dict):
    """Store search history entry."""
    if text not in d:
        d[text] = [(doc_name, tf_idf)]
    else:
        d[text].append((doc_name, tf_idf))
    return d


def run(text: str):
    main()
    result = query(text.lower())
    if result is None:
        return ["No result."]
    return_list = []
    remaining = len(result)
    for i in result:
        s = f"{i[2]} - {i[0]}"
        return_list.append(s)
        if remaining != 1:
            s += "\n"
            remaining -= 1
    return return_list


if __name__ == "__main__":
    main()
    text = input("Search: ").strip().lower()
    result = query(text)
    if result is not None:
        for doc_name, word, tf_idf in result:
            print(doc_name, round(tf_idf, 4))
    else:
        print("No Result.")
