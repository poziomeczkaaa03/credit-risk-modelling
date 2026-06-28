# Credit Risk Modelling

Projekt dedykowany klasyfikacji klientów banku pod kątem ryzyka niespłacenia zobowiązania kredytowego (defaultu). Celem analizy było stworzenie optymalnego modelu predictive-scoringowego na podstawie danych finansowych, historii kredytowej oraz parametrów wnioskowanego kredytu.

## Główne Hipotezy Badawcze
1. Klienci o wyższym stosunku kwoty kredytu do dochodu charakteryzują się wyższym prawdopodobieństwem defaultu.
2. Historia wcześniejszego defaultu oraz niższa ocena kredytowa (rating) klienta są jednymi z najsilniejszych predyktorów ryzyka niespłacenia zobowiązania.

## Przetwarzanie i Czyszczenie Danych
Początkowy zbiór danych (32 581 obserwacji, 12 atrybutów) został poddany rygorystycznemu procesowi czyszczenia, w wyniku którego wyeliminowano błędy i obserwacje odstające. Ostateczny zbiór danych objął 28 495 obserwacji:
* Usunięto 165 zduplikowanych wpisów.
* Usunięto braki danych w zmiennych `person_emp_length` oraz `loan_int_rate`.
* Wyeliminowano anomalie i outliery: wiek > 100 lat, dochód > 2.5 mln, staż pracy > 50 lat oraz staż pracy przekraczający wiek (wiek - 15 lat).

## Zastosowane Modele i Metodologia
W projekcie zaimplementowano i porównano cztery modele uczenia maszynowego (z wykorzystaniem walidacji krzyżowej 5-Fold CV oraz optymalizacji hiperparametrów `GridSearchCV`):
1. **Regresja Logistyczna** (z regularyzacją ElasticNet/Lasso i korektą wag dla niezbalansowanych klas).
2. **Drzewo Decyzyjne** (poddane procedurze przycinania `ccp_alpha` w celu uniknięcia przeuczenia).
3. **Las Losowy (Random Forest)**.
4. **Maszyna Wektorów Nośnych (SVM)** z jądrem RBF.

## Wyniki i Ewaluacja Modelu
* **Random Forest** osiągnął najwyższą ogólną zdolność dyskryminacyjną ze wskaźnikiem **AUC = 0.9353**, najwyższą dokładnością (Accuracy = 0.9340) oraz najlepszymi wskaźnikami Lift i Gain.
* **Regresja Logistyczna** charakteryzowała się najwyższą czułością (**Recall = 0.7786**), co jest kluczowe w maksymalizacji wykrywalności klientów zagrożonych niewypłacalnością.

## Analiza Biznesowa (Zysk z Modeli)
Wprowadzono symulację finansową uwzględniającą parametry ryzyka:
* **EAD (Exposure at Default):** pełna kwota kredytu.
* **LGD (Loss Given Default):** uzależnione od statusu własności nieruchomości (od 20% dla kredytów hipotecznych do 65% dla pozostałych).
* **Przychód:** zebrana marża odsetkowa.

Modele zoptymalizowano pod kątem maksymalizacji czystego zysku netto portfela testowego. Najwyższy wynik finansowy dla banku wygenerował model **Random Forest (maksymalizacja zysku do poziomu 4 979 825.49 USD)** przy optymalnym punkcie odcięcia (Cut-Off) na poziomie 38.96%.

## Struktura Repozytorium
* `cr_loan2.csv` – zbiór danych wejściowych.
* `Credit_risk.pdf` – prezentacja z wynikami i agendą biznesową.
* `credit_risk_project.ipynb` – Jupyter Notebook zawierający pełen kod (EDA, czyszczenie danych, modelowanie, GridSearch, wizualizacje krzywych ROC i analizę finansową).

## Wymagania (Python)
* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `seaborn`
