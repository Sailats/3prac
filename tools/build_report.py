"""Создаёт читаемый отчёт и материалы для защиты из реального JSON-протокола."""
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'handwritten': 'Рукописный', 'printed': 'Печатный', 'unknown': 'Отказ',
          'invalid': 'Неверный формат', 'error': 'Сбой'}


def percent(value):
    return f'{value * 100:.1f}%'.replace('.', ',')


def main():
    report = json.loads((ROOT / 'results/latest_test.json').read_text(encoding='utf-8'))
    model_names = {key: value['name'] for key, value in report['models'].items()}
    compared_models = ' и '.join(model_names.values())
    prompt_blocks = ''.join(
        f'<h3>{html.escape(model_names[key])}</h3><pre>{html.escape(prompt)}</pre>'
        for key, prompt in report['prompts'].items())
    summary_rows = []
    for key, metrics in report['metrics'].items():
        hw = metrics['classes']['handwritten']
        values = [model_names[key], percent(metrics['accuracy']), percent(hw['precision']),
                  percent(hw['recall']), percent(hw['f1']), percent(metrics['macro_f1']),
                  percent(metrics['coverage']), f'{metrics["mean_seconds"]:.2f} с',
                  str(metrics['unknown']), str(metrics['invalid']), str(metrics['errors'])]
        summary_rows.append('<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>')
    summary = '<div class="scroll"><table><tr>' + ''.join(
        f'<th>{title}</th>' for title in ('Модель', 'Accuracy', 'Precision¹', 'Recall¹', 'F1¹',
                                       'F1 macro', 'Coverage', 'Среднее время', 'Отказ', 'Формат', 'Сбой'))
    summary += '</tr>' + ''.join(summary_rows) + '</table></div>'
    matrices = ''
    for key, metrics in report['metrics'].items():
        matrices += f'<h3>{model_names[key]}</h3><div class="scroll"><table><tr><th>Факт / ответ</th>'
        outcomes = ('handwritten', 'printed', 'unknown', 'invalid', 'error')
        matrices += ''.join(f'<th>{LABELS[outcome]}</th>' for outcome in outcomes) + '</tr>'
        for label in ('handwritten', 'printed'):
            matrices += f'<tr><th>{LABELS[label]}</th>' + ''.join(
                f'<td>{metrics["confusion"][label][outcome]}</td>' for outcome in outcomes) + '</tr>'
        matrices += '</table></div>'
    best = max(report['metrics'], key=lambda key: report['metrics'][key]['accuracy'])
    fastest = min(report['metrics'], key=lambda key: report['metrics'][key]['mean_seconds'])
    conclusion = (f'На этой выборке лучшая Accuracy у {model_names[best]}: '
                  f'{percent(report["metrics"][best]["accuracy"])}. '
                  f'Меньшее среднее время у {model_names[fastest]}: '
                  f'{report["metrics"][fastest]["mean_seconds"]:.2f} с. '
                  'Сравнение относится к данным и параметрам этого эксперимента, а не ко всем задачам VLM.')
    examples = []
    errors = [item for item in report['predictions'] if item['label'] != item['truth']]
    correct = [item for item in report['predictions'] if item['label'] == item['truth']]
    for item in (correct[:1] + errors[:2]):
        data = base64.b64encode((ROOT / 'data' / item['file']).read_bytes()).decode()
        examples.append(f'<figure><img src="data:image/jpeg;base64,{data}" alt="Пример {item["id"]}">'
                        f'<figcaption>{item["id"]}, {model_names[item["model"]]}: '
                        f'факт — {LABELS[item["truth"]]}, ответ — {LABELS[item["label"]]}. '
                        f'Сырой ответ: <code>{html.escape(item["raw"])}</code></figcaption></figure>')
    body = f'''
<p><a href="/">Открыть программу</a></p>
<h1>Сравнение мультимодальных моделей в задаче определения типа текста</h1>
<p>Практическая работа по дисциплине «Интеллектуальные системы и технологии».</p>
<div class="callout"><strong>Способ запуска.</strong> Две модели с Hugging Face выполняются
локально без токенов и платных запросов. В исходном обсуждении слайда указаны
Inference Providers; локальный запуск заменяет именно этот пункт задания.</div>
<h2>1. Цель и постановка задачи</h2>
<p>По изображению определить, написан ли текст человеком от руки или напечатан.
Сравнить {compared_models} по качеству и времени. Пользователь загружает
фотографию, получает два ответа и может проверить модели на размеченной выборке.</p>
<h2>2. Данные</h2>
<p>Использованы 50 реальных фрагментов фотографий и сканов из
<a href="https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys">rukopys</a>
(Ukrainian Catholic University и участники набора, CC BY 4.0): 25 рукописных и
25 печатных. Пять изображений каждого класса образуют dev, остальные двадцать — test.
Каждый фрагмент получен из отдельного исходного документа. Авторы почерка не
идентифицированы; разные документы могут принадлежать одному автору.</p>
<p>Отбор выполнялся по исходной разметке и геометрии, затем проверялся визуально
до эксперимента. Похожие водяные знаки и неоднозначные фрагменты исключены.
Оставлен один водяной знак и несколько сложных узких/размытых фрагментов.
Происхождение, координаты, исключения и хеши записаны в data/provenance.json
и data/manifest.csv. Исходная разметка может содержать ошибки.</p>
<h2>3. Метод и воспроизводимость</h2>
<p>Модели предобучены; дополнительного обучения нет (zero-shot). На вход поступают
только пиксели и инструкция для выбранной модели. Имя файла и правильный класс не
передаются. Изображение переводится в RGB и уменьшается с сохранением пропорций
до максимальной стороны 1024 пикселя. Далее применяются штатные процессоры моделей;
для обеих моделей установлен предел 512 визуальных токенов. Линейные слои
языкового декодера сжаты до NF4 с двойным квантованием; визуальный модуль и
выходной слой сохраняют FP16. Квантование выполнялось без обучения и без
использования разметки. Это позволяет держать обе модели в памяти видеокарты.</p>
{prompt_blocks}
<p>Генерация детерминированная, до 12 новых токенов. Разрешены handwritten, printed,
unknown. Другой ответ считается неверным форматом, ошибка исполнения — сбоем.
Даже если в неверном ответе есть описание текста, класс по нему не угадывается.
В предыдущей версии SmolVLM 500M показала слабый результат и была заменена
на Qwen3-VL 2B. Формулировки выбраны на 10 dev-примерах: подробная для Qwen2-VL,
короткая для Qwen3-VL. Классы и задача одинаковы, текст инструкций различается.
Это сравнение настроенных программных связок, а не изолированная оценка моделей
при идентичной формулировке. Разметка сохранена. Эти 40 изображений уже
использовались при выборе моделей, поэтому числа служат повторным учебным
сравнением; для независимой оценки нужна новая выборка. История сохранена в results/.</p>
<p>Перед измерением каждая модель прогревается на одном dev-изображении. Загрузка
и прогрев исключены из тестового времени, порядок моделей чередуется. Время включает
подготовку входа, генерацию и декодирование. Это не универсальная оценка скорости.
Дата UTC: {html.escape(report['created_utc'])}. ПК: {html.escape(str(report['environment']))}.</p>
<h2>4. Метрики и фактический результат</h2>
<p>Accuracy = число правильных ответов / все 40 изображений. Precision = TP/(TP+FP),
Recall = TP/(TP+FN), F1 = 2PR/(P+R). Отказы и ошибки уменьшают Recall и Accuracy.
Coverage — доля ответов одного из двух классов. F1 macro — среднее F1 двух классов.
При нулевом знаменателе показатель принимается равным нулю.</p>
{summary}<p>¹ Для класса «Рукописный». Показатели обоих классов есть в JSON.</p>
<h2>5. Матрицы ошибок</h2>{matrices}
<h2>6. Примеры и ошибки</h2>{''.join(examples)}
<p>Ошибки могут возникать на декоративном печатном шрифте, курсиве, слабом контрасте
и обрезанных строках. Иногда модель переписывает текст вместо классификации —
это отдельная измеряемая проблема следования инструкции.</p>
<h2>7. Вывод</h2><p>{conclusion}</p>
<p>Одна ошибка на test меняет Accuracy на 2,5 процентного пункта. Набор небольшой,
не проверены новые авторы почерка, полные страницы, любые языки и произвольные сцены.
Нельзя обещать такую же точность на других фотографиях. Смешанные страницы и пустые
кадры формально допускают unknown, но их отдельная оценка в эти 40 примеров не входит.</p>
<h2>8. Что разработано</h2>
<p>predict.py — подготовка фото и model.generate; evaluate.py — эксперимент и метрики;
app.py — локальный сервер и фоновые задачи; web/ — интерфейс; download_models.py —
общедоступные веса; setup_windows.ps1 — запуск Windows. Программа не распознаёт класс
по имени файла и не использует удалённый сервис. Веса и библиотеки уже входят
в полный переносимый архив.</p>
<h2>9. Протокол и источники</h2>
<p>Полный протокол: {html.escape(report['report_file'])}, CSV с ответами, latest_test.json.
SHA256 набора инструкций (JSON с sort_keys=True): <code>{report['prompts_sha256']}</code>.
SHA256 разметки: <code>{report['manifest_sha256']}</code>.</p>
<ul><li><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">Qwen2-VL 2B</a>, Apache 2.0.</li>
<li><a href="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct">Qwen3-VL 2B</a>, Apache 2.0.</li>
<li><a href="https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys">rukopys</a>, CC BY 4.0,
фрагменты обрезаны, уменьшены, сохранены с белым полем.</li></ul>
'''
    css = '''body{font:16px/1.65 "Segoe UI",Arial,sans-serif;color:#172c40;max-width:1080px;margin:40px auto;padding:0 25px}h1{font:38px/1.2 Georgia,serif}h2{margin-top:35px}a{color:#245d9d}.callout{background:#fff3da;padding:16px;border-left:3px solid #bd8c37}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;border:1px solid #d9e1e9}th{background:#edf2f7}.scroll{overflow:auto}pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 Consolas,monospace}pre{padding:16px;background:#f2f5f8}img{max-width:100%;max-height:200px}figure{padding:15px;border:1px solid #d9e1e9;margin:15px 0}figcaption{font-size:13px;margin-top:8px}@media print{body{margin:0;padding:0;font-size:11pt}h2{break-after:avoid}figure,table{break-inside:avoid}h1{font-size:26pt}.callout{background:white}}'''
    standalone = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Отчёт — классификация текста</title><style>' + css + '</style><body>' + body.replace('<p><a href="/">Открыть программу</a></p>', '') + '</body></html>'
    (ROOT / 'docs/Отчет.html').write_text(standalone, encoding='utf-8')
    guide = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Материалы для защиты</title><link rel="stylesheet" href="/web/style.css"><body><main class="guide">' + body + '</main></body></html>'
    (ROOT / 'web/guide.html').write_text(guide, encoding='utf-8')
    speech = f'''ВЫСТУПЛЕНИЕ НА 2–3 МИНУТЫ

Мой проект определяет тип текста на изображении: рукописный или печатный.
Практический пример применения — предварительная сортировка документов.

Я сравнил две готовые мультимодальные модели: {compared_models}.
Обе получают изображение и инструкцию, а выдают текстовый ответ.
Формулировки подобраны отдельно на dev; задача и классы совпадают.
Они запускаются на компьютере. Ключ и платный сервис не используются.
Собственное обучение не выполнялось: это zero-shot классификация.

В проекте 50 реальных фрагментов из открытого rukopys, лицензия CC BY 4.0.
10 использовались для разработки и проверки инструкции. Сравнительная выборка —
40 изображений, поровну двух классов. В новой версии выполнено повторное
сравнение на том же наборе; это не новая независимая проверка.
Документы между dev и test не повторяются.
Разметка и имена файлов не передаются нейросетям.

Я разработал подготовку изображений, вызовы моделей, интерфейс, процедуру
сравнения, расчёт метрик и сохранение протоколов. Сейчас покажу фотографию
и ответы моделей. Затем покажу сохранённый эксперимент и матрицы ошибок.

{conclusion}

Отказы, неверный формат и ошибки не исключались из Accuracy. Поэтому показатель
не завышается отбрасыванием неудачных запросов. Исходные ответы сохранены.
Основное ограничение — малая выборка: одна ошибка меняет точность на 2,5 пункта.

ВОПРОСЫ ПРЕПОДАВАТЕЛЯ

Где интеллектуальная часть?
В предобученных VLM: визуальный энкодер и языковая модель обрабатывают пиксели
и инструкцию. Реальный вызов — model.generate в predict.py.

Почему обе модели имеют размер 2B?
Они помещаются на этом ПК и относятся к разным поколениям Qwen-VL.
Сравнивается качество двух настроенных связок. Нельзя объяснять разницу
только поколением: архитектуры, обучение и формулировки инструкций различаются.

Почему инструкции отличаются?
На dev подробная инструкция лучше работала с Qwen2, а короткая — с Qwen3.
После этого формулировки зафиксированы и выполнено сравнение на 40 изображениях.
Поэтому в отчёте показаны обе инструкции и названы ограничения сравнения.

Вы обучали нейросеть?
Нет. Использованы готовые модели без дообучения. Разработано приложение и эксперимент.

Почему нет «уверенности 97%»?
Сгенерированное моделью число не является откалиброванной вероятностью.
Мы показываем класс, время и исходный ответ, а качество оцениваем на разметке.

Что такое Precision, Recall и F1?
Precision показывает долю верных рукописных ответов среди всех ответов «рукописный».
Recall — какую долю всех рукописных примеров нашли. F1 объединяет эти показатели.
Macro F1 — среднее F1 рукописного и печатного классов.

Зачем квантование NF4?
Чтобы обе модели одновременно помещались в 8 ГБ видеопамяти. Сжаты линейные
слои языкового декодера; визуальная часть и выходной слой оставлены в FP16.
Обучения на тестовых примерах при этом не было. Качество измерено после сжатия.

Зачем заменили SmolVLM?
Она дала 25% на сравнительной выборке и часто нарушала формат ответа.
Новая модель выбрана для более надёжной демонстрации. Старый результат сохранён.

Что с требованием Hugging Face Providers?
Веса взяты с Hugging Face, но выполняются локально. Это осознанная замена
платного способа вызова. Если требуются строго облачные провайдеры, этот пункт
отличается от исходного слайда — нельзя утверждать полное совпадение.

Можно ли считать тест независимым по авторам почерка?
Нет. Документы разные, но идентичность авторов в исходных данных не проверена.

ЧТО ПОКАЗЫВАТЬ
1. Запустить Start_Windows.bat.
2. Готовый рукописный и печатный примеры, затем собственную фотографию.
3. Таблицу 40 изображений, матрицы, реальный ошибочный ответ.
4. Раскрыть «Ответ нейросети», открыть CSV или JSON протокола.
5. При вопросе о коде показать predict.py, evaluate.py и app.py.
'''
    (ROOT / 'docs/Защита.txt').write_text(speech, encoding='utf-8-sig')
    print(conclusion)


if __name__ == '__main__':
    main()
