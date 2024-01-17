### Temat
Wykorzystanie uczenia semi-nadzorowanego w sieciach prototypowych w zadaniach klasyfikacji muzyki

### Zespół
- Olga Krupa
- Adam Górski

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

## Dokumentacja końcowa
### Zaimplementowane modele
#### Sieć prototypowa
W ramach projektu zastosowaliśmy prototypową sieć neuronową, która jest specjalnie przeznaczona do rozwiązywania problemów uczenia się na małych zbiorach danych. Sieć ta opiera swoje działanie na metodzie meta-uczenia, wykorzystując metryki jako narzędzie do oceny jej skuteczności. 

Proces szkolenia sieci neuronowej dla metod meta-uczenia opartych na metryce wykorzystuje metodę treningu epizodycznego. Trening epizodyczny polega na tworzeniu epizodów, w których wybieramy określoną liczbę $K$ klas ze zbioru klas oraz przykładów dla każdej z nich z głównego zbioru treningowego, tworząc tym samym zbiór wsparcia oraz losowe dane dla każdej klasy, które tworzą zbiór zapytań. Błąd, który jest obliczany dla danego epizodu, jest wykorzystywany do aktualizacji parametrów wag sieci.

Celem treningu epizodycznego jest stworzenie epizodów, które naśladują problem uczenia na małych zbiorach danych poprzez selekcję podpróbki klas i danych, tworząc dla każdej iteracji epizod jako indywidualne zadanie. W trakcie treningu model jest stale zmuszony do radzenia sobie z nowymi wyzwaniami klasyfikacji K-shot, N-way.

Sieć prototypowa jest siecią neuronową opartą na metryce wykorzystywaną w uczeniu z małą liczbą przykładów. Sieci te wykorzystują przestrzeń osadzenia oraz skupianie się punktów należącej do tej samej klasy blisko siebie. Na podstawie zbioru wspierającego uczymy sieć osadzania tych punktów w przestrzeni, a następnie dla każdej klasy obliczamy **prototyp**. Prototyp danej klasy jest średnią z osadzeń danych ze zbioru wsparcia. Klasyfikacji dokonujemy przez wyszukania najbliższego prototypu dla osadzenia punku etykietowanego.

Prototyp jest średnią wszystkich osadzeń danych ze zbioru wsparcia dla danej klasy $k \in K$. Funkcję osadzenia z możliwymi do nauczenia parametrami $\phi$ oznaczamy jako $f_\phi$. Prototyp $c$ dla danej klasy $k$ obliczamy następująco:

$$c_k = \frac{\sum_{i} f_\phi(x_i) z_{i,k}}{\sum_{i} z_{i,k}}, \quad \text{gdzie} \quad z_{i,k} = \mathbf{1}[y_i = k]$$

 Podobieństwo danej próbki wejściowej $X$ obliczamy za pomocą kwadratowej odległości euklidesowej $d$ między osadzeniem próbki a prototypem dla każdej klasy $k \in K$.
$$P(y = k | X) = \frac{\exp(-d(f_\phi(X),c_k))}{\sum_{k \in K} \exp(-d(f_\phi(X),c_k))}$$

#### Trening sieci prototypowej
W celu wytrenowania sieci prototypowej należy stworzyć _DataLoader_ dla wybranego zbioru danych oraz zdefiniować `model_type: 'protonet'` w pliku `config/dataset.yml`. W pliku `config/dataset.yml` możemy również określić wartości dla:
* sample_rate: 16000
* n_way: 3 - Liczba klas do próbkowania dla epizodu.
* n_support: 5 - Liczba próbek na klasę do wykorzystania jako zbiór wsparcia.
* n_query: 20 - Liczba próbek na klasę do wykorzystania jako zbiór zapytań.
* n_unlabeled: 5 - Liczba próbek na klasę do wykorzystania jako dane nieoznakowane.
* n_distractor: 2 - Liczba klas dystraktorów.
* n_train_episodes: 100 - Liczba epizodów.
* n_val_episodes: 50 - Liczba epizodów walidacji. 
* num_workers: 10

Po zdefiniowaniu modelu należy wywołać komendę `make dataset*`. 

Po uzyskaniu DataLoadera dla wybranego zbioru danych należy również zdefiniować `model_type: 'protonet'` w pliku `config/training.yml`. W pliku `config/training.yml` określane są również parametry:
* sample_rate: 16000
* max_epochs: 10 - Maksymalna liczba epizodów podczas treningu.
* log_every_n_steps: 10 - Jak często dodawać wiersze logowania.
* val_check_interval: 5 - Jak często sprawdzać zestaw walidacyjny.
* profiler: "simple"
* logger: "wandb"
* wand_project_name: "wimu-agorski-okrupa"
* model_output_path: "models/irmas"
 
Po zdefiniowaniu modelu oraz opcjonalnie dodatkowych argumentów należy wywołać komendę `make train`. 

#### Few-shot learning dla podejścia uczenia częściowo nadzorowanergo
W podstawowym podejściu dla sieci prototypowych wykorzystywane są dwa zbiory do treningu epizodycznego: zbiór wsparcia $S$ (support set) oraz zbiór zapytań $Q$ (query set). Dla podejścia Few-shot learning częściowo nadzorowanego zbiór uczący jest oznaczany jako krotka oznaczonych i nieoznaczonych przykładów: ($S$, $R$). Oznaczona część jest zwykłym zbiorem wsparcia $S$, zawierającym listę krotek wejść i klas. Oprócz zbioru wsparcia został wprowadzony nieoznakowany zbiór $R$ zawierający tylko wejścia - $R$ (bez podania klas do których dany przykład należy). Podczas treningu epizodycznego wykorzystujemy dane, które składają się z zestawu wsparcia $S$, zestawu nieoznakowanych danych $R$ i zestawu zapytań $Q$. Celem tego podejścia jest wykorzystanie etykietowanych elementów w zbiorze $S$ oraz nieoznakowanych elementów w zbiorze $R$ w każdym epizodzie, aby osiągnąć dobrą wydajność w odpowiadającym zestawie zapytań. Elementy nieoznakowane w zbiorze $R$ mogą być istotne dla rozważanych klas lub mogą być elementami rozpraszającymi, które należą do klasy nieistotnej dla bieżącego epizodu. Należy jednak zauważyć, że model nie posiada rzeczywistych informacji na temat tego, czy każdy nieoznakowany przykład jest elementem rozpraszającym czy nie. 

##### Sieć prototypowa z soft k-means
Podejście te czerpie inspirację z częściowo nadzorowanego klastrowania. Podejście soft k-means zakłada niejawnie, że każdy nieoznakowany przykład należy do jednej z $K$ klas w danym epizodzie. Patrząc na każdy prototyp jako środek klastra, proces udoskonalania może próbować dostosować lokalizację klastrów, aby lepiej pasowały do przykładów zarówno ze zbioru wspomagającego, jak i danych nieoznakowanych. W tej perspektywie przyporządkowania klastrów etykietowanych przykładów ze zbioru wspomagającego są uznawane za znane i ustalone na podstawie etykiety każdego przykładu. Proces udoskonalania musi z kolei oszacować przyporządkowania klastrów dla nieoznakowanych przykładów i dostosować lokalizację klastrów (prototypów) odpowiednio. 
W ramach algorytmu tworzymy prototypy dla zestawu wsparcia $S$ jako lokalizacji klastrów.


$$c_k = \frac{\sum_{i} f_\phi(x_i) z_{i,k}}{\sum_{i} z_{i,k}} , \quad gdzie \quad z_{i,k} = \mathbf{1} [y_i = k ]$$

Następnie, nieoznakowane przykłady otrzymują częściowe przyporządkowanie $z_{j,k}^{\`}$ do każdego klastra na podstawie odległości euklidesowej od lokalizacji klastrów.

$$z_{j,k}^{\`} = \frac{\exp(-d(f_\phi(x_j^{\`}),c_k))}{\sum_{k'} \exp(-d(f_\phi(x_j^{\`}),c_k'))}$$


Ostatecznie, uzyskuje się udoskonalone prototypy poprzez uwzględnienie 
tych nieoznakowanych przykładów. 
$$c_k^{\`} = \frac{\sum_{i} f_\phi(x_i) z_{i,k} + \sum_{j} f_\phi(x_j^{\`}) z_{j,k}^{\`}}{\sum_{i} z_{i,k} + \sum_{j} z_{j,k}^{\`}}$$

##### Trening sieci prototypowej z soft k-means
Podobnie jak dla sieci prototypowej należy stworzyć _DataLoader_ dla wybranego zbioru danych oraz zdefiniować `model_type: 'softkmeans'` w pliku `config/dataset.yml`. W pliku `config/dataset.yml` należy również określić wartość dla `n_unlabeled`, czyli liczbę próbek na klasę do wykorzystania jako dane nieoznakowane.

Po zdefiniowaniu modelu należy wywołać komendę `make dataset*`. 

Po uzyskaniu DataLoadera dla wybranego zbioru danych należy również zdefiniowaniu `model_type: 'softkmeans'` w pliku `config/training.yml`. 

Po zdefiniowaniu modelu oraz opcjonalnie dodatkowych argumentów należy wywołać komendę `make train`. 

##### Sieć prototypowa z soft k-means z klasą dystraktorów
Ponieważ podejście soft k-means opisane powyżej zakłada niejawnie, że każdy nieoznakowany przykład należy do jednej z $K$ klas w danym epizodzie, powstała potrzeba stworzenia podejścia w którym tworzony jest model odpornyna istnienie przykładów z innych klas, które nazywamy klasami zakłócającymi (distractor classes). Ponieważ algorytm soft k-means rozkłada swoje miękkie przyporządkowania na wszystkie klasy, dane zakłócające mogą być szkodliwe i zakłócać proces udoskonalania, ponieważ prototypy zostaną dostosowane, aby częściowo uwzględnić również te zakłócające przykłady. Prostym sposobem na rozwiązanie tego problemu jest dodanie dodatkowego klastra, którego celem jest przechwytywanie 
zakłócających przykładów, uniemożliwiając im zanieczyszczanie klastrów klas interesujących.

$$
c_k = \begin{cases} 
\frac{\sum_{i} f_\phi(x_i) z_{i,k}}{\sum_{i} z_{i,k}} & \text{for } k = 1 \ldots K \\
0 & \text{for } k = K + 1
\end{cases}
$$

Przyjmujemy upraszczające założenie, że klaster dystraktorów ma prototyp wyśrodkowany na początku. Rozważamy również wprowadzenie skali długości $r_k$ do reprezentowania zmian w odległościach wewnątrz klastra odległości, szczególnie dla klastra rozpraszającego.

$$z_{j,k}^{\`} = \frac{\exp(-\frac{1}{r_k^2}d(f_\phi(x_j^{\`}),c_k) - A(r_k)}{\sum_{k'} \exp(-\frac{1}{r_k^2}d(f_\phi(x_j^{\`}),c_k') - A(r_k^{\`})},  \quad \text{gdzie} \quad  A(r_k^{\`}) = \frac{1}{2} log(2 \pi) + log(r)$$

Ostatecznie, uzyskujemy udoskonalone prototypy poprzez uwzględnienie 
tych nieoznakowanych przykładów tak jak zostało to opisane w sieci prototypowej z soft k-means. 

##### Trening sieci prototypowej z soft k-means z klasą dystraktorów
Podobnie jak dla wcześniejszych sieci należy stworzyć _DataLoader_ dla wybranego zbioru danych oraz zdefiniować `model_type: 'softkmeansdistractor'` w pliku `config/dataset.yml`. W pliku `config/dataset.yml` należy również określić wartość dla `n_unlabeled` oraz `n_distractor`, czyli liczbę klas dystraktorów.

Po zdefiniowaniu modelu należy wywołać komendę `make dataset*`. 

Po uzyskaniu DataLoadera dla wybranego zbioru danych należy również zdefiniowaniu `model_type: 'softkmeansdistractor'` w pliku `config/training.yml`. 

Po zdefiniowaniu modelu oraz opcjonalnie dodatkowych argumentów należy wywołać komendę `make train`.


##### Spełnienie wymagań technologicznych
Modele zostały zaimplementowane z użyciem Pytorch Lightninga tak jak było w opisywanym przez nas rozwiązaniu.
Dodatakowo udało nam się też zintegrować proces uczenia ze śledzeniem i logowaniem przy pomocy tensorboarda
albo weights and biases. Kod w pythonie przestrzega dobre praktyki - korzysta z walidacji przy pomocy pydantic-a.
Dodatakowo kluczowe wartości związane z przetwarzaniem danych, trenowaniem i wdrożeniem są zawarte w plikach
config/dataset.yml, config/deployment.yml i training.yml co pozwala na uproszczoną pracę nad rozwiązaniem.
Dodaliśmy także aplikację w streamlicie oraz wdrożenie modelu przy pomocy fastapi.

##### Testowanie wdrożenia
Kod zajmujący się testowaniem wdrożenia znajduje się w folderze tests/e2e/test_e2e.py. Żeby mógł zadziałać poprawnie,
użytkownik musi wcześniej wykonać instrukcję
```bash
make deploy
``` 
a następnie przetestować wdrożenie przy pomocy
```
make e2e_test
```

##### Dokumentacja API

###### Endpoint: /predict
Ten punkt końcowy API przyjmuje zbiór wsparcia (support) i zbiór zapytań (query) jako wejście, wykonuje wnioskowanie przy użyciu określonego modelu i zwraca wynik predykcji.

###### Parametry żądania:
- support (typ: SupportModel): Model Pydantic reprezentujący zbiór wsparcia, który zawiera dane audio, etykiety docelowe i listę nazw klas.
- query (typ: QueryModel): Model Pydantic reprezentujący zbiór zapytań, który zawiera dane audio, dla których mają zostać wykonane predykcje.
###### Odpowiedź:
Odpowiedź API to obiekt JSON zawierający następujące pola:

- logits (typ: List[List[float]]): Surowe wyniki z modelu dla każdego próbkowania z zapytania.
- predicted_labels (typ: List[int]): Przewidziane etykiety odpowiadające próbkom z zapytania.
- predicted_classes (typ: List[str]): Przewidziane nazwy klas odpowiadające próbkom z zapytania.

###### Modele danych
- SupportModel
Ten model Pydantic reprezentuje zbiór wsparcia, zapewniając, że dane wejściowe spełniają określone wymagania.

audio (typ: List[List[List[float]]]): Lista 3D reprezentująca dane audio dla każdej próbki w zbiorze wsparcia.
target (typ: List[int]): Lista etykiet docelowych odpowiadających każdej próbce w zbiorze wsparcia.
classlist (typ: List[str]): Lista nazw klas odpowiadających etykietom docelowym.
Walidacja:
Dane audio są walidowane, aby sprawdzić, czy mają oczekiwany format (lista 3D).
Dane target są walidowane, aby zapewnić, że wszystkie klasy docelowe mają takie same rozkłady.

- QueryModel 
Ten model Pydantic reprezentuje zbiór zapytań, zapewniając, że dane wejściowe spełniają określone wymagania.

audio (typ: List[List[List[float]]]): Lista 3D reprezentująca dane audio dla każdej próbki w zbiorze zapytań.
Walidacja:
Dane audio są walidowane, aby sprawdzić, czy mają oczekiwany format (lista 3D).
PredictOutput
Ten model Pydantic reprezentuje wynik A### Temat
Wykorzystanie uczenia semi-nadzorowanego w sieciach prototypowych w zadaniach klasyfikacji muzyki

### Zespół
- Olga Krupa
- Adam Górski

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

## Dokumentacja końcowa
### Zaimplementowane modele
#### Sieć prototypowa
W ramach projektu zastosowaliśmy prototypową sieć neuronową, która jest specjalnie przeznaczona do rozwiązywania problemów uczenia się na małych zbiorach danych. Sieć ta opiera swoje działanie na metodzie meta-uczenia, wykorzystując metryki jako narzędzie do oceny jej skuteczności. 

Proces szkolenia sieci neuronowej dla metod meta-uczenia opartych na metryce wykorzystuje metodę treningu epizodycznego. Trening epizodyczny polega na tworzeniu epizodów, w których wybieramy określoną liczbę $K$ klas ze zbioru klas oraz przykładów dla każdej z nich z głównego zbioru treningowego, tworząc tym samym zbiór wsparcia oraz losowe dane dla każdej klasy, które tworzą zbiór zapytań. Błąd, który jest obliczany dla danego epizodu, jest wykorzystywany do aktualizacji parametrów wag sieci.

Celem treningu epizodycznego jest stworzenie epizodów, które naśladują problem uczenia na małych zbiorach danych poprzez selekcję podpróbki klas i danych, tworząc dla każdej iteracji epizod jako indywidualne zadanie. W trakcie treningu model jest stale zmuszony do radzenia sobie z nowymi wyzwaniami klasyfikacji K-shot, N-way.

Sieć prototypowa jest siecią neuronową opartą na metryce wykorzystywaną w uczeniu z małą liczbą przykładów. Sieci te wykorzystują przestrzeń osadzenia oraz skupianie się punktów należącej do tej samej klasy blisko siebie. Na podstawie zbioru wspierającego uczymy sieć osadzania tych punktów w przestrzeni, a następnie dla każdej klasy obliczamy **prototyp**. Prototyp danej klasy jest średnią z osadzeń danych ze zbioru wsparcia. Klasyfikacji dokonujemy przez wyszukania najbliższego prototypu dla osadzenia punku etykietowanego.

Prototyp jest średnią wszystkich osadzeń danych ze zbioru wsparcia dla danej klasy $k \in K$. Funkcję osadzenia z możliwymi do nauczenia parametrami $\phi$ oznaczamy jako $f_\phi$. Prototyp $c$ dla danej klasy $k$ obliczamy następująco:

$$c_k = \frac{\sum_{i} f_\phi(x_i) z_{i,k}}{\sum_{i} z_{i,k}}, \quad \text{gdzie} \quad z_{i,k} = \mathbf{1}[y_i = k]$$

 Podobieństwo danej próbki wejściowej $X$ obliczamy za pomocą kwadratowej odległości euklidesowej $d$ między osadzeniem próbki a prototypem dla każdej klasy $k \in K$.
$$P(y = k | X) = \frac{\exp(-d(f_\phi(X),c_k))}{\sum_{k \in K} \exp(-d(f_\phi(X),c_k))}$$

#### Trening sieci prototypowej
W celu wytrenowania sieci prototypowej należy stworzyć _DataLoader_ dla wybranego zbioru danych oraz zdefiniować `model_type: 'protonet'` w pliku `config/dataset.yml`. W pliku `config/dataset.yml` możemy również określić wartości dla:
* sample_rate: 16000
* n_way: 3 - Liczba klas do próbkowania dla epizodu.
* n_support: 5 - Liczba próbek na klasę do wykorzystania jako zbiór wsparcia.
* n_query: 20 - Liczba próbek na klasę do wykorzystania jako zbiór zapytań.
* n_unlabeled: 5 - Liczba próbek na klasę do wykorzystania jako dane nieoznakowane.
* n_distractor: 2 - Liczba klas dystraktorów.
* n_train_episodes: 100 - Liczba epizodów.
* n_val_episodes: 50 - Liczba epizodów walidacji. 
* num_workers: 10

Po zdefiniowaniu modelu należy wywołać komendę `make dataset*`. 

Po uzyskaniu DataLoadera dla wybranego zbioru danych należy również zdefiniować `model_type: 'protonet'` w pliku `config/training.yml`. W pliku `config/training.yml` określane są również parametry:
* sample_rate: 16000
* max_epochs: 10 - Maksymalna liczba epizodów podczas treningu.
* log_every_n_steps: 10 - Jak często dodawać wiersze logowania.
* val_check_interval: 5 - Jak często sprawdzać zestaw walidacyjny.
* profiler: "simple"
* logger: "wandb"
* wand_project_name: "wimu-agorski-okrupa"
* model_output_path: "models/irmas"
 
Po zdefiniowaniu modelu oraz opcjonalnie dodatkowych argumentów należy wywołać komendę `make train`. 

#### Few-shot learning dla podejścia uczenia częściowo nadzorowanergo
W podstawowym podejściu dla sieci prototypowych wykorzystywane są dwa zbiory do treningu epizodycznego: zbiór wsparcia $S$ (support set) oraz zbiór zapytań $Q$ (query set). Dla podejścia Few-shot learning częściowo nadzorowanego zbiór uczący jest oznaczany jako krotka oznaczonych i nieoznaczonych przykładów: ($S$, $R$). Oznaczona część jest zwykłym zbiorem wsparcia $S$, zawierającym listę krotek wejść i klas. Oprócz zbioru wsparcia został wprowadzony nieoznakowany zbiór $R$ zawierający tylko wejścia - $R$ (bez podania klas do których dany przykład należy). Podczas treningu epizodycznego wykorzystujemy dane, które składają się z zestawu wsparcia $S$, zestawu nieoznakowanych danych $R$ i zestawu zapytań $Q$. Celem tego podejścia jest wykorzystanie etykietowanych elementów w zbiorze $S$ oraz nieoznakowanych elementów w zbiorze $R$ w każdym epizodzie, aby osiągnąć dobrą wydajność w odpowiadającym zestawie zapytań. Elementy nieoznakowane w zbiorze $R$ mogą być istotne dla rozważanych klas lub mogą być elementami rozpraszającymi, które należą do klasy nieistotnej dla bieżącego epizodu. Należy jednak zauważyć, że model nie posiada rzeczywistych informacji na temat tego, czy każdy nieoznakowany przykład jest elementem rozpraszającym czy nie. 

##### Sieć prototypowa z soft k-means
Podejście te czerpie inspirację z częściowo nadzorowanego klastrowania. Podejście soft k-means zakłada niejawnie, że każdy nieoznakowany przykład należy do jednej z $K$ klas w danym epizodzie. Patrząc na każdy prototyp jako środek klastra, proces udoskonalania może próbować dostosować lokalizację klastrów, aby lepiej pasowały do przykładów zarówno ze zbioru wspomagającego, jak i danych nieoznakowanych. W tej perspektywie przyporządkowania klastrów etykietowanych przykładów ze zbioru wspomagającego są uznawane za znane i ustalone na podstawie etykiety każdego przykładu. Proces udoskonalania musi z kolei oszacować przyporządkowania klastrów dla nieoznakowanych przykładów i dostosować lokalizację klastrów (prototypów) odpowiednio. 
W ramach algorytmu tworzymy prototypy dla zestawu wsparcia $S$ jako lokalizacji klastrów.


$$c_k = \frac{\sum_{i} f_\phi(x_i) z_{i,k}}{\sum_{i} z_{i,k}} , \quad gdzie \quad z_{i,k} = \mathbf{1} [y_i = k ]$$

Następnie, nieoznakowane przykłady otrzymują częściowe przyporządkowanie $z_{j,k}^{\`}$ do każdego klastra na podstawie odległości euklidesowej od lokalizacji klastrów.

$$z_{j,k}^{\`} = \frac{\exp(-d(f_\phi(x_j^{\`}),c_k))}{\sum_{k'} \exp(-d(f_\phi(x_j^{\`}),c_k'))}$$


Ostatecznie, uzyskuje się udoskonalone prototypy poprzez uwzględnienie 
tych nieoznakowanych przykładów. 
$$c_k^{\`} = \frac{\sum_{i} f_\phi(x_i) z_{i,k} + \sum_{j} f_\phi(x_j^{\`}) z_{j,k}^{\`}}{\sum_{i} z_{i,k} + \sum_{j} z_{j,k}^{\`}}$$

##### Trening sieci prototypowej z soft k-means
Podobnie jak dla sieci prototypowej należy stworzyć _DataLoader_ dla wybranego zbioru danych oraz zdefiniować `model_type: 'softkmeans'` w pliku `config/dataset.yml`. W pliku `config/dataset.yml` należy również określić wartość dla `n_unlabeled`, czyli liczbę próbek na klasę do wykorzystania jako dane nieoznakowane.

Po zdefiniowaniu modelu należy wywołać komendę `make dataset*`. 

Po uzyskaniu DataLoadera dla wybranego zbioru danych należy również zdefiniowaniu `model_type: 'softkmeans'` w pliku `config/training.yml`. 

Po zdefiniowaniu modelu oraz opcjonalnie dodatkowych argumentów należy wywołać komendę `make train`. 

##### Sieć prototypowa z soft k-means z klasą dystraktorów
Ponieważ podejście soft k-means opisane powyżej zakłada niejawnie, że każdy nieoznakowany przykład należy do jednej z $K$ klas w danym epizodzie, powstała potrzeba stworzenia podejścia w którym tworzony jest model odpornyna istnienie przykładów z innych klas, które nazywamy klasami zakłócającymi (distractor classes). Ponieważ algorytm soft k-means rozkłada swoje miękkie przyporządkowania na wszystkie klasy, dane zakłócające mogą być szkodliwe i zakłócać proces udoskonalania, ponieważ prototypy zostaną dostosowane, aby częściowo uwzględnić również te zakłócające przykłady. Prostym sposobem na rozwiązanie tego problemu jest dodanie dodatkowego klastra, którego celem jest przechwytywanie 
zakłócających przykładów, uniemożliwiając im zanieczyszczanie klastrów klas interesujących.

$$
c_k = \begin{cases} 
\frac{\sum_{i} f_\phi(x_i) z_{i,k}}{\sum_{i} z_{i,k}} & \text{for } k = 1 \ldots K \\
0 & \text{for } k = K + 1
\end{cases}
$$

Przyjmujemy upraszczające założenie, że klaster dystraktorów ma prototyp wyśrodkowany na początku. Rozważamy również wprowadzenie skali długości $r_k$ do reprezentowania zmian w odległościach wewnątrz klastra odległości, szczególnie dla klastra rozpraszającego.

$$z_{j,k}^{\`} = \frac{\exp(-\frac{1}{r_k^2}d(f_\phi(x_j^{\`}),c_k) - A(r_k)}{\sum_{k'} \exp(-\frac{1}{r_k^2}d(f_\phi(x_j^{\`}),c_k') - A(r_k^{\`})},  \quad \text{gdzie} \quad  A(r_k^{\`}) = \frac{1}{2} log(2 \pi) + log(r)$$

Ostatecznie, uzyskujemy udoskonalone prototypy poprzez uwzględnienie 
tych nieoznakowanych przykładów tak jak zostało to opisane w sieci prototypowej z soft k-means. 

##### Trening sieci prototypowej z soft k-means z klasą dystraktorów
Podobnie jak dla wcześniejszych sieci należy stworzyć _DataLoader_ dla wybranego zbioru danych oraz zdefiniować `model_type: 'softkmeansdistractor'` w pliku `config/dataset.yml`. W pliku `config/dataset.yml` należy również określić wartość dla `n_unlabeled` oraz `n_distractor`, czyli liczbę klas dystraktorów.

Po zdefiniowaniu modelu należy wywołać komendę `make dataset*`. 

Po uzyskaniu DataLoadera dla wybranego zbioru danych należy również zdefiniowaniu `model_type: 'softkmeansdistractor'` w pliku `config/training.yml`. 

Po zdefiniowaniu modelu oraz opcjonalnie dodatkowych argumentów należy wywołać komendę `make train`.


##### Spełnienie wymagań technologicznych
Modele zostały zaimplementowane z użyciem Pytorch Lightninga tak jak było w opisywanym przez nas rozwiązaniu.
Dodatakowo udało nam się też zintegrować proces uczenia ze śledzeniem i logowaniem przy pomocy tensorboarda
albo weights and biases. Kod w pythonie przestrzega dobre praktyki - korzysta z walidacji przy pomocy pydantic-a.
Dodatakowo kluczowe wartości związane z przetwarzaniem danych, trenowaniem i wdrożeniem są zawarte w plikach
config/dataset.yml, config/deployment.yml i training.yml co pozwala na uproszczoną pracę nad rozwiązaniem.
Dodaliśmy także aplikację w streamlicie oraz wdrożenie modelu przy pomocy fastapi.

##### Testowanie wdrożenia
Kod zajmujący się testowaniem wdrożenia znajduje się w folderze tests/e2e/test_e2e.py. Żeby mógł zadziałać poprawnie,
użytkownik musi wcześniej wykonać instrukcję
```bash
make deploy
``` 
a następnie przetestować wdrożenie przy pomocy
```
make e2e_test
```

##### Dokumentacja API

###### Endpoint: /predict
Ten punkt końcowy API przyjmuje zbiór wsparcia (support) i zbiór zapytań (query) jako wejście, wykonuje wnioskowanie przy użyciu określonego modelu i zwraca wynik predykcji.

###### Parametry żądania:
- support (typ: SupportModel): Model Pydantic reprezentujący zbiór wsparcia, który zawiera dane audio, etykiety docelowe i listę nazw klas.
- query (typ: QueryModel): Model Pydantic reprezentujący zbiór zapytań, który zawiera dane audio, dla których mają zostać wykonane predykcje.
###### Odpowiedź:
Odpowiedź API to obiekt JSON zawierający następujące pola:

- logits (typ: List[List[float]]): Surowe wyniki z modelu dla każdego próbkowania z zapytania.
- predicted_labels (typ: List[int]): Przewidziane etykiety odpowiadające próbkom z zapytania.
- predicted_classes (typ: List[str]): Przewidziane nazwy klas odpowiadające próbkom z zapytania.

###### Modele danych
- SupportModel
Ten model Pydantic reprezentuje zbiór wsparcia, zapewniając, że dane wejściowe spełniają określone wymagania.

audio (typ: List[List[List[float]]]): Lista 3D reprezentująca dane audio dla każdej próbki w zbiorze wsparcia.
target (typ: List[int]): Lista etykiet docelowych odpowiadających każdej próbce w zbiorze wsparcia.
classlist (typ: List[str]): Lista nazw klas odpowiadających etykietom docelowym.
Walidacja:
Dane audio są walidowane, aby sprawdzić, czy mają oczekiwany format (lista 3D).
Dane target są walidowane, aby zapewnić, że wszystkie klasy docelowe mają takie same rozkłady.

- QueryModel 
Ten model Pydantic reprezentuje zbiór zapytań, zapewniając, że dane wejściowe spełniają określone wymagania.

audio (typ: List[List[List[float]]]): Lista 3D reprezentująca dane audio dla każdej próbki w zbiorze zapytań.
Walidacja:
Dane audio są walidowane, aby sprawdzić, czy mają oczekiwany format (lista 3D).
PredictOutput
Ten model Pydantic reprezentuje wynik API.

logits (typ: List[List[float]]): Surowe wyniki z modelu dla każdego próbkowania z zapytania.
predicted_labels (typ: List[int]): Przewidziane etykiety odpowiadające próbkom z zapytania.
predicted_classes (typ: List[str]): Przewidziane nazwy klas odpowiadające próbkom z zapytania.

###### MLOps

Główną platformą z której korzystaliśmy do moniotorowania modelu było "weights and biases". Wyniki można obserwować
tutaj - https://wandb.ai/adamsebastiangorski/wimu-agorski-okrupa?workspace=user-adamsebastiangorski. Żeby mógł
Pan zobaczyć rezultaty potrzebowalibyśmy Pana maila. Weights and Biases integruje się bardzo dobrze z pytorch-lightningiem.
