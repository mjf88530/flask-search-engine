# flask-search-engine
A tiny text search engine built with **Flask** and **scikit-learn**.  
It indexes a folder of `.txt` documents, builds a bag-of-words model, and returns the top matches for a query.

## Quickstart

```bash
# create & activate a venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# install deps
pip install -r requirements.txt

# run the app
cd app
python demo.py
```
Open http://127.0.0.1:5000 and try a search.

> macOS/Linux:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> cd app && python demo.py
> ```

## Project layout
```
app/
demo.py # Flask app
searcher.py # indexing + search logic
templates/ # HTML templates
large-sample/ # small sample corpus (200 .txt files)
document.csv # doc -> size (bytes)
terms.csv # term -> document frequency
```

## Data
```
- Use the included `app/large-sample/` to run quickly.
- To use more data, drop `.txt` files into `app/large-sample/` and run once; the index builds automatically.
- Large artifacts like `app/index.dat` are intentionally not committed.
```
## Notes
```
- Dev server only (Flask debug).
- If you change the corpus folder, update `DATABASE_PATH` in `app/searcher.py`.
```

