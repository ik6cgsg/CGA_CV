# CV

## Task

Предположим что мы с вами разрабатываем инструмент проверки подлинности бумажной банкноты. Для идентификации банкноты используются многоугольники произвольной формы без самопересечений. Необходимо реализовать программу производящую поиск всех контрольных меток на изоражении. На вход программа принимает изображение, а также список базисных многоугольников. В ответе необходимо указать количество многоугольников, для каждого многоугольника указать его номер во входном списке, центр и угол поворота.

На вход программе подается число N, а далее на N последующих строках заданы координаты X, Y - отрезков задающих многоугольник. Гарантируется что многоугольники несамопересекающиеся, а также замкнуты.  А также изображение разрешением 600 на 200 пикселей. 

На выход программа должна выводить число M - количество обнаруженных примитивов, а на следующих M строках - номер фигуры, смещение х, смещение у, масштаб и угол поворота. 

**Оценка - балл за лабораторную - значение IoU * 1.1**

### P.S.

* Гарантируется что все примитивы изображены линией толщиной в 1 пиксель.
* Точность результата метрике IoU не должна быть меньше чем 70%
* Проверка будет осуществлять путем рендеринга представленных вами факторов фигуры и сравниваться по метрике IoU. Таким образом в случае квадрата не имеет значения какой вы вывели угол поворота 45градусов или 135.
* При проверке будет использоваться следующая последовательность движений:
  1. Масштабирование
  2. Поворот
  3. Сдвиг 
* Гарантируется что все целевые фигуры полностью лежат внутри изображения
* Датасет не предоставляется. Тестирующее множество будет состоять из 100 автоматически сгенерированных изображений.
* Гарантируется что на предоставленных изображениях метод Хафа в 4х мерном пространстве с “модификациями” (это секрет, но без всяких нейронок) - дает IoU 91%
* **Кто побьет метрику 95% - получает тотальный почет, уважение, и личнй пакет защиты и поддержки на любую тему по трудоустройству, собеседованиям, стажировкам и так далее от Савчука Даниила, не более чем 30 минут в неделю до 30.06.2020.**

