# Ultralyzer

A pipeline to analyze retinal UltraWidefield images using MVC GUI structure and background processing.

## Project Structure

```text
Ultralyzer/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── ultralyzer/
│       ├── __init__.py
│       ├── apps/
│       │   ├── __init__.py
│       │   ├── app_factory.py
│       │   └── app.py
│       ├── controllers/
│       │   ├── __init__.py
│       │   └── main_controller.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── image_model.py
│       ├── views/
│       │   ├── __init__.py
│       │   └── main_view.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── db_utils.py
├── tests/
│   ├── __init__.py
│   └── test_app.py
└── requirements.txt
```  

## Modules

- `apps`: Application entrypoints and factory.
- `controllers`: GUI controllers.
- `models`: Data models (e.g. SQLite database interactions).
- `views`: GUI layouts.
- `utils`: Utility functions (e.g. image processing, DB utils).
