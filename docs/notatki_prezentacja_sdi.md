## Tytułowy (1)
Nazywam sie Kamil Cisek...

## Plan prezentacji (2)
Plan prezentacji

## Tomografia komputerowa (3)
Aby przedstawić temat pracy na początek warto zapoznać się z tym, czym jest tomografia komputerowa. Nazwa "tomografia" wywodzi się od greckich słów oznaczających "przekrój" i "zapisywać". Jest to technika rejestracji przekroju na podstawie pomiarów dokonywanych z zewnątrz, w sposób nieinwazyjny. Czyli mamy jakiś obiekt, np. przekrój ciała człowieka i na podstawie pewnych pomiarów z zewnątrz chcemy poznać wnętrze obiektu bez potrzeby naruszania go. 

Pomiar polega na rejestracji osłabiania promieniowania X przez obiekt, aby zrekonstruować obraz dokonujemy dużo takich pojedynczych pomiarów. Następnie na podstawie tych danych rekonstruuje się obraz przekroju numerycznie. Na zdjęciu współczesny tomograf komputerowy.

## Problem prosty i odwrotny (4)
W zagadnieniu tomografii wyłaniają się nam 2 problemy. Problem prosty, czyli pytanie jakie pomiary osłabiania promieniowania (projekcje) g dostaniemy jeśli prześwietlimy obiekt f. Aby odpowiedzieć na to pytanie musimy znać transformację R. W tomografii komputerowej transformacja ta jest nazywana transformacją Radona.

Jeśli jest problem prosty jest też problem odwrotny, jaki obraz f dostaniemy z pomiarów g, gdy zastosujemy transformację odwrotną R^{-1}, w tomografii odwrotną transformację Radona. Jest to podstawowe zagadnienia tomografii komputerowej, ponieważ mamy dostęp do projekcji a nie mamy dostępu do wnętrza obiektu.

## Problem prosty i odwrotny (5)
Aby zrekonstruować przekrój obiektu na podstawie pomiarów tomograficznych należy dokonać transformacji odwrotnej.
Nie jest to jednak problem prosty numerycznie, jest to problem nieliniowy, stąd wiele podejść do zadania transformacji odwrotnej, w tym zastosowanie do tego celu uczenia maszynowego, co jest tematem tej pracy.

Na obrazku obiekt, w tym przypadku mózg człowieka i projekcje - sinogramy. Dokładniej o tym co przedstawia obrazek projekcji opowiem za chwilę.

## Cel pracy (6)
* Generacja syntetycznego zbioru uczącego do treningu - syntetyczny w sensie część jego będzie generowane, w tym przypadku obrazy dostępne, generowane sinogramy. Decyzja o generacji zbioru syntetycznego zapadła ponieważ nie ma dobrych, dużych i ogólnodostępnych zbiorów danych z zarówno przekrojami tomograficznymi jak i projekcjami. Postać projekcji jest zależna od budowy tomografu, więc wielu producentów raczej chroni te dane.
* Zastosowanie uczenia maszynowego (z nadzorem) do rozwiązania problemu odwrotnego w tomografii komputerowej.
* Określenie minimalnego rozmiaru zbioru uczącego potrzebnego do osiągnięcia dobrych wyników.
* Sprawdzenie, czy możliwe jest nauczenie transformacji odwrotnej przy użyciu zbioru syntetycznego wygenerowanego na naturalnych obrazach (nie tomograficznych). Będzie to pewien eksperyment mający sprawdzić na ile w uczeniu do rozwiązywania problemu odwrotnego gra rolę adaptacja dziedzinowa, więcej o tym opowiem później.
* Opcjonalnie, porównanie wyników z klasycznymi metodami rekonstrukcji. Jest to punkt opcjonalny, ponieważ uzyskanie lepszych wyników niż algorytmy klasyczne za pomocą uczenia maszynowego może być trudne i wykraczać poza zakres pracy inżynierskiej. Poza tym ogranicza nas jakość syntetycznego zbioru danych, który będzie generowany za pomocą jakiejść liniowej aproksymacji.

## Projekcje tomograficzne (7)
Teraz trochę o fizyce projekcji tomograficznych. Każda projekcja to zbiór pomiarów osłabienia promieniowania X pod określonym kątem z określonego miejsca. Zacząć robić rysunek...

Wzór ten mówi, że gdy promień rentgenowski przechodzi przez materiał, jego natężenie maleje wykładniczo w zależności od tego, jak silnie dany materiał go pochłania. W tomografii komputerowej każdy promień „widzi” tylko sumaryczne tłumienie na swojej drodze — czyli całkę z μ(x,y).

## Projekcje tomograficzne (8)
Jest kilka podstawowych geometrii skanerów tomograficznych, w pracy zajmę się geometrią równoległą skanera i dla takiej będzie generowany zbiór uczący. A więc chcemy przeprowadzić wiele pomiarów aby zrekonstruować obraz tomograficzny. Tomograf składa się z tuby gdzie leży człowiek, po jednej stronie znajdują się źródła promieniowania a po drugiej detektory. Współczesne tomografy wykonują wiele pomiarów jednocześnie, po czym źródła i detektory obracają się o pewien kąt.

(Teraz rysunek na tablicy podobny jak rysunek 1)
Na rysunku po lewej stronie widzimy przekrój ciała człowieka. Jasność pikseli odpowiada skali Hounsfielda, która mówi o tym ile promieniowania pochłania dany piksel w stosunku do wody. Np. kości pochłaniają więcej niż woda a powietrze mniej.

Z tych pomiarów powstaje sinogram jak na rysunku 2. Każdy pasek w osi X to skany wykonane w jednym momencie przy jednej pozycji detektorów. Jasność piksela oznacza ilość pochłoniętego promieniowania przy danym pomiarze, im piksel jaśniejszy tym więcej. Nazwa sinogram wywodzi się z tego, że z powodu ruchu źródeł i detektorów po okręgu na tak stworzonym obrazie widać różne funkcje sinusoidalne, na tym obrazku jest przedstawiony sinogram pojedynczego punktu - czysty sinus.

## Zbiór uczący - założenia (9)
Punkt po punkcie...
Problem jest wymiarowo duży, potrzebny będzie dużo przykładów uczących, duży model. Dlatego będzie potrzebne dużo miejsca na zbiór uczący (rząd 1TB) i szybka karta graficzna z duża ilością VRAM. Na szczęście są takie w instytucie promotora.

## Metody rekonstrukcji obrazów tomograficznych (10-11)
Chciałbym teraz przedstawić podział metod rekonstrukcji obrazów tomograficznych.

Pierwsze to metody analityczne, bazują na transformacji odwrotnej Radona. Najczęściej stosowany algorytm to algorytm FBP (Filtered Back Projection - Filtrowanej projekcji wstecznej). 

Składa się on z dwóch części: filtrowania i projekcji wstecznej.
Najpierw wytłumaczę projekcję wsteczną...
Jako, że jedynie takie postępowanie powoduje powstawanie rozmytych obrazów wcześniej aplikuje się filtr na sinogramie, który ma na celu odszumienie sinogramu.

Jest to algorytm najczęściej stosowany obecnie w rzeczywistych tomografach, jako że to szybki i względnie prosty algorytm. Ma on jednak tendencje do tworzenia na obrazie niechcianych artefaktów i szumów (szczególnie przy małej dawce promieniowania), jest to jego główny problem, który popycha badaczy do poszukiwania ulepszeń i innych metod.

## Metody rekonstrukcji obrazów tomograficznych (12)
Następna grupa metod to metody statystyczne. Są one stosunkowo rzadko stosowane w praktyce, jednak chciałbym o nich powiedzieć. 

Metody statystyczne zakładają, że dane pomiarowe w tomografii mają charakter losowy, np. zgodny z rozkładem Poissona. Rekonstrukcja polega na znalezieniu obrazu, który maksymalizuje prawdopodobieństwo uzyskania zmierzonych danych.Przykładem takiego podejścia jest algorytm ML-EM (Maximum Likelihood – Expectation Maximization), który iteracyjnie poprawia obraz, dopasowując go do danych pomiarowych.

## Metody rekonstrukcji obrazów tomograficznych (13)
Następna grupa metod to metody algebraiczne.

W tych metodach patrzymy na proces obrazowania jako układ równań liniowych. Poszukujemy rozwiązania x - nieznanych wartości tłumienia w pikselach obrazu na podstawie sinogramy y, mając układ równań, który opisuje macierz A. Celem jest znalezenia jak najlepiej dopasowanego x.

Aby rozwiązać ten układ stosuje się iteracyjne algorytmy optymalizacyjne. Metody iteracyjne polegają na znalezieniu współczynnika pochłaniania poszczególnych wokselach poprzez kolejne próby modyfikacji tych współczynników tak, aby ich wartości zgadzały się ze zmierzonymi. 

Metody iteracyjne są bardziej złożone obliczeniowo niż algorytm FBP, ale lepiej radzi sobie z niepełnymi i zaszumionymi danymi i wytwarza mniej artefaktów.

## Metody rekonstrukcji obrazów tomograficznych (14)
Teraz dochodzimy do zastosowanie uczenia maszynowego, które będzie tematej tej pracy.

Chcemy wykorzystać głębokie sieci neuronowe do nauki transformacji odwrotnej. Naszą nadzieją i motywacją jest to, że mimo czasochłonnego treningu wytrenowana sieć będzie generować obrazy tomograficzne szybko, zredukuje ilość szumów i artefaktów w porównaniu do klasycznych metod, a także da lepszą jakość rekonstrukcji przy niskich dawkach promieniowania.

Na rysunku schemat działania...

## Architektura sieci neuronowej (15)
Najczęściej proponowaną w literaturze siecią do tego zadania jest sieć U-Net.

U-Net (nazwa od wyglądu architektury jak literka U) jest konwolucyjną siecią neuronową złożoną z kodera i dekodera i połączęń "skip" pomiedzy warstwami. W części kodera konwolucyjne warstwy zmniejszają rozmiar obrazka a w dekoderze stopniowo przywracają początkowy rozmiar. Dzięki bottleneckowi na środku sieć ekstrahuje cechy obrazu.

Pozwala skutecznie łączyć informacje globalne i lokalne, dzięki czemu doskonale sprawdza się w zadaniach rekonstrukcji i segmentacji obrazów medycznych.

## Architektura sieci neuronowej (16)
Innym pomysłem na rekonstrukcję obrazów tomografii jest zastosowanie sieci GAN (Generative Adversarial Network). Jest to model generatywny składajacy się z generatora, który generuje obrazy jak najabardziej podobne do tych, na których był trenowany i dyskryminatora, który ma wykrywać fałszywe obrazy (nie ze zbioru treningowego) wygenerowane przez generator.

Odmianą sieci GAN do transformacji jednego obrazu na drugi jest model Pix2Pix. Jest to GAN warunkowany, więc można mu podać obraz wejściowy na podstawie, którego ma wygenerować wyjście. Pierwotnym celem Pix2Pix jest generacja obrazu w jednym stylu na podstawie obrazu w drugim stylu, np. zdjęcie w nocy na podstawie zdjęcia w dzień, mapa na podstawie zdjęcia satelitarnego albo jak na rysunku obraz na podstawie szkicu. Jednak po pewnych przekształceniach można spróbować zastosować go do rekonstrukcji obrazów tomograficznych.

## Architektura sieci neuronowej (17)
Następnym pomysłem na architekturę sieci neuronowej jest transformer. Możnaby eksperymentalnie przekształcić transformer wizyjny, który na swoje wejście przyjmuje obraz podzielony na tokeny (w naszym przypadku przyjmowałby sinogram), aby jako sekwencję wyjściową generował obraz tomograficzny. W literaturze znalazłem jedynie zastosowania transformera do poprawy jakości sinogramów i obrazów tomograficznych, nie znalazłem zastosowania transformera w zadaniu generacji end-to-end, więc takie użycie byłoby nowością, co jest zaletą. Jednak to eksperymentalne podejście, nie wiadomo czy się powiedzie.

Istnieje ponadto opcja kombinacji kilku przedstawionych wcześniej architektur modeli. Bardzo możliwe, że taka opcja zostanie wdrożona, ponieważ w uczeniu maszynowym połączenie zalet kilku architektur jest często siłą. Przykładem takiego podejścia może być np. zastosowanie U-Neta do podstawowej rekonstrukcji, a model Pix2Pix do odszumiania wyniku U-Neta jako, że jest dobry w przekształcaniu pomiędzy obrazami z podobną strukturą przestrzenną.

## Trening sieci neuronowej (18)
Punkt po punkcie...

## Eksperyment z naturalnymi obrazami (19)
W treningu sieci chciałbym zrobić eksperyment ze zbiorem danych wygenerowanych na podstawie obrazów naturalnych, nie tomograficznych. Do tego celu trzeba będzie wygenerować drugi zbiór danych, być może mniejszy od zbioru podstawowego ze względów praktycznych i wytrenować sieć na takich obrazach. Następnie zamierzam porównać rezultaty w rekonstrukcji obrazów tomograficznych z siecią trenowaną na podstawowym zbiorze na podobnej ilości przykładów. Zbiór walidacyjny dla obu sieci to zbió© wygenerowany na obrazach tomograficznych.

Eksperyment ten pozwoli sprawdzić, czy sieć może się nauczyć ogólnej transformacji odwrotnej niezależnie od specyfiki obrazów. Problemem może być zjawisko nazywane w uczeniu maszynowym adaptacją dziedzinową (domain adaptation). Polaga ono na tym, że sieć neuronowa ma tendencję do słabszej generalizacji dla danych innych niż dane uczące. Ocena wielkości tego zjawiska w przypadku zadania pracy będzie celem eksperymentu.

## Metryki walidacyjne jakości rekonstrukcji (20)
MSE mierzy średni kwadrat różnicy między wartościami pikseli obrazu oryginalnego i zrekonstruowanego.
Im mniejsza wartość MSE, tym mniejsze są błędy rekonstrukcji i tym bardziej obraz przypomina oryginał.
Metryka jest prosta i powszechnie stosowana, ale nie zawsze dobrze oddaje subiektywną jakość wizualną obrazu.

SSIM ocenia podobieństwo struktury, kontrastu i jasności pomiędzy dwoma obrazami.
Wartość bliska 1 oznacza niemal identyczne obrazy, natomiast niższe wartości wskazują na utratę szczegółów lub zmianę kontrastu. Jest bardziej zgodny z ludzkim postrzeganiem jakości niż MSE.

PSNR wyraża jakość rekonstrukcji w decybelach (dB), porównując maksymalną możliwą wartość sygnału do błędu MSE.
Wyższa wartość PSNR oznacza lepszą jakość obrazu, zwykle powyżej 30 dB uznaje się za bardzo dobrą rekonstrukcję.
Metryka ta jest często stosowana w przetwarzaniu obrazów i kompresji.

CC mierzy siłę liniowej zależności pomiędzy pikselami oryginalnego i odtworzonego obrazu.
Wartość 1 oznacza idealną korelację, 0 - brak związku, a -1 - zależność odwrotną.
Dzięki temu metryka dobrze pokazuje, czy zachowany został ogólny kształt i rozkład struktur w obrazie.

Do całościowej oceny jakości rekonstrukcji będą stosowane wszystkie metryki i porównywane relacje między nimi. Ponadto porównywany będzie także rozkład metryk w zbiorze walidacyjnym aby ocenić czy nie ma części zbioru treningowego, dla których wartości metryk znacznie się różnią od reszty. Porównywane będzie także odchylenie stanadardowe metryk.

## Dotychczasowe wyniki prac (21)
Punkt po punkcie...

## Dotychczasowe wyniki prac (22)
Wykres przedstawia błąd treningowy i testowy w trakcie treningu sieci U-Net. Zastosowano mechanizm wczesnego zatrzymania w momencie gdy strata za biorze testowym zaczyna rosnąć.

## Dotychczasowe wyniki prac (23)
Jak widać obrazy są zaszumione, jest to główny element do poprawy...

## Plan bieżącego semestru (24)
Punkt po punkcie...