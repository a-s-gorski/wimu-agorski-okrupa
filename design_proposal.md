### Temat
Wykorzystanie uczenia semi-nadzorowanego w sieciach prototypowych w zadaniach klasyfikacji muzyki

### Harmonogram
- 27.10-14.11 - dokładne zapoznanie się z artykułami naukowymi, konfiguracja środowiska, pierwszy prototyp i eksperymenty
- 15.11-31.11 - optymalizacja hiperparameterów przy pomocy optuny, analiza błędów i efektywności przy pomocy weight and biases.
- 01.12-31.12 - analiza wyników, początek pisania artykułu naukowego.
- 01.01-14.01 - dalsze konsultacje, analiza wyników i postępów

### Bibliografia
W ramach projektu zostanie wykorzystana sieć prototypowa, której działanie jest opisane w materiale naukowym
[Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175). Powyższa sieć jest wykorzystywana w zadaniach few-shot learning, czyli uczenia na zbiorze z małą ilością przykładów trenujących.
Głównym celem projektu jest sprawdzenie oraz zaimplementowanie różnych metod semi-nadzorowanych w sieciach prototypowych. Badania te będą się opierać na pracy naukowej [Meta-Learning for Semi-Supervised Few-Shot Classification](https://arxiv.org/abs/1803.00676).
Dodatkowo planujemy zaimplementować nową metodę MUSIC, której działanie jest opisane w artykule [An Embarrassingly Simple Approach to Semi-Supervised Few-Shot Learning](https://arxiv.org/abs/2209.13777). 
Będziemy się także opierać na [Few-Shot and Zero-Shot Learning for Music Information Retrieval](https://music-fsl-zsl.github.io/tutorial) jako merytorycznym fundamencie w kwestiach "few-shot learning" w dziedzinie MIR.

### Zakres eksperymentów
- Przetestowanie wykorzystywanych bibliotek i efektywności prototypu.
- Zbadnie różnych metod uczenia semi-nadzorowanego w sieciach prototypowych.
- Optymalizacja hiperparameterów przy pomocy optun-y.
- Przetestowanie efektywności modelu z punktu widzenia użytkownika przy pomocy aplikacji webowej.

### Funkcjonalnosć programu
- Kod, który pozwala na automatyczną instalację bibliotek oraz instalację paczki przy pomocy poetry.
- Skrypty pozwalające na automatyczne trenowanie i walidację modelu z zadanymi parameterami. 
- Aplikacja webowa w streamlit-cie pozwalająca na przetestowanie wytrenowanych przez nas modeli do klasyfikacji załadowanych przez nas próbek.
- Najlepszy model będzie także wdrożony automatycznie przy pomocy fastapi i Dockera.

### Planowany stack technologiczny
- Python
- Pytorch Lightning,
- Weight and Biases,
- Tenserboard,
- Music-fsl
- Optuna
- Poetry
- Unittests, PyTest - w razie potrzeby także Hypothesis
- argparse, click
- logging
- Docker
- fastapi
- flake8, autopep8
- cookiecutter based project structure
