'use strict';
const $ = id => document.getElementById(id);
const names = {handwritten:'Рукописный',printed:'Печатный',unknown:'Не удалось определить',invalid:'Неверный формат ответа',error:'Ошибка запуска'};
const modelNames = {'2B':'Qwen2-VL 2B','Q3':'Qwen3-VL 2B'};
let selected = null, busy = false, previewUrl = null, polling = false;
const pct = value => (value * 100).toLocaleString('ru-RU', {maximumFractionDigits:1}) + '%';
const seconds = value => value.toLocaleString('ru-RU', {maximumFractionDigits:2}) + ' с';
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function setBusy(value) {
  busy = value;
  $('recognize').disabled = value || !selected;
  $('evaluate').disabled = value;
  $('file').disabled = value;
  $('clear').disabled = value;
  document.querySelectorAll('.example').forEach(button => button.disabled = value);
  $('progress-panel').hidden = !value;
}
function resetAnswers() {
  for (const key of ['2B','Q3']) {
    $('value-'+key).textContent = '—'; $('value-'+key).className = 'answer-value';
    $('detail-'+key).textContent = 'Готова к сравнению'; $('raw-box-'+key).hidden = true;
  }
  $('agreement').textContent = 'Готово к сравнению'; $('agreement').className = 'badge';
}
function showImage(url, name) {
  $('preview').src = url; $('preview').hidden = false; $('upload-prompt').hidden = true;
  $('filename').textContent = name; $('clear').hidden = false;
  error(''); resetAnswers(); setBusy(false);
}
function selectFile(file) {
  if (!file || busy) return;
  if (file.size > 15*1024*1024) return error('Файл больше 15 МБ. Выберите изображение поменьше.');
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file); selected = {file};
  document.querySelectorAll('.example').forEach(button => button.classList.remove('selected'));
  showImage(previewUrl, file.name);
}
$('file').addEventListener('change', event => selectFile(event.target.files[0]));
$('clear').addEventListener('click', () => {
  selected = null; $('file').value = ''; $('preview').hidden = true;
  $('upload-prompt').hidden = false; $('clear').hidden = true;
  $('filename').textContent = 'Фотография ещё не выбрана';
  document.querySelectorAll('.example').forEach(button => button.classList.remove('selected'));
  resetAnswers(); setBusy(false);
});
for (const name of ['dragenter','dragover']) $('dropzone').addEventListener(name, event => {
  event.preventDefault(); if (!busy) $('dropzone').classList.add('drag');
});
for (const name of ['dragleave','drop']) $('dropzone').addEventListener(name, event => {
  event.preventDefault(); $('dropzone').classList.remove('drag');
  if (name === 'drop') selectFile(event.dataTransfer.files[0]);
});
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Не удалось выполнить запрос.');
  return data;
}
async function start(path, body) {
  error(''); setBusy(true); $('progress').value = 0; $('progress-message').textContent = 'Подготовка…';
  try { await api(path, body); poll(); }
  catch (exc) { error(exc.message); setBusy(false); }
}
$('recognize').addEventListener('click', async () => {
  if (!selected || busy) return;
  if (selected.sample_id) return start('/api/predict', {sample_id:selected.sample_id});
  const reader = new FileReader();
  reader.onload = () => start('/api/predict', {image:reader.result.split(',')[1]});
  reader.onerror = () => error('Не удалось прочитать файл. Выберите его ещё раз.');
  reader.readAsDataURL(selected.file);
});
$('evaluate').addEventListener('click', () => start('/api/evaluate', {split:'test'}));
async function poll() {
  if (polling) return;
  polling = true;
  try {
    while (true) {
      const status = await api('/api/status');
      $('progress-message').textContent = status.message;
      $('progress').max = status.total || 2; $('progress').value = status.done;
      if (status.status !== 'running') {
        if (status.status === 'error') error(status.message);
        if (status.status === 'done' && status.result) {
          if (status.result.kind === 'prediction') showAnswers(status.result.predictions);
          if (status.result.kind === 'evaluation') showReport(status.result.report);
        }
        setBusy(false); break;
      }
      setBusy(true); await new Promise(resolve => setTimeout(resolve, 900));
    }
  } catch (exc) { error('Соединение с программой потеряно. Проверьте, что окно запуска открыто.'); setBusy(false); }
  finally { polling = false; }
}
function showAnswers(predictions) {
  for (const item of predictions) {
    $('value-'+item.model).textContent = names[item.label] || item.label;
    $('value-'+item.model).className = 'answer-value' + (['printed','handwritten'].includes(item.label) ? ' good' : '');
    $('detail-'+item.model).textContent = item.error || `${seconds(item.seconds)} · ${item.device === 'cuda' ? 'видеокарта' : 'процессор'}` + (item.load_seconds > 0 ? ' · включая загрузку модели' : '');
    $('raw-'+item.model).textContent = item.raw || item.error || 'Пустой ответ';
    $('raw-box-'+item.model).hidden = false;
  }
  const agree = predictions.every(item => ['printed','handwritten'].includes(item.label)) && predictions[0].label === predictions[1].label;
  $('agreement').textContent = agree ? 'Модели согласны' : 'Ответы различаются или есть отказ';
  $('agreement').className = 'badge' + (agree ? ' agree' : '');
}
function cell(row, text, tag='td', className='') {
  const element = document.createElement(tag); element.textContent = text;
  element.className = className; row.append(element); return element;
}
function showReport(report) {
  const entries = Object.entries(report.metrics || {});
  $('metrics').replaceChildren(); $('matrix').replaceChildren(); $('report-links').replaceChildren();
  if (!entries.length) { $('report-caption').textContent = 'Сохранённых результатов пока нет. Запустите эксперимент.'; return; }
  const date = new Date(report.created_utc).toLocaleString('ru-RU');
  $('report-caption').textContent = `Реальный запуск: ${date}. ${report.environment?.gpu || 'CPU'}. Загрузка и прогрев исключены из времени.`;
  const scroll = document.createElement('div'); scroll.className = 'table-scroll';
  const table = document.createElement('table'), head = table.createTHead().insertRow();
  for (const title of ['Модель','Accuracy','Precision¹','Recall¹','F1¹','F1 macro','Определено','Среднее время']) cell(head,title,'th');
  const body = table.createTBody();
  for (const [key,m] of entries) {
    const row = body.insertRow();
    const hw = m.classes.handwritten;
    for (const value of [modelNames[key]||key,pct(m.accuracy),pct(hw.precision),pct(hw.recall),pct(hw.f1),pct(m.macro_f1),pct(m.coverage),seconds(m.mean_seconds)]) cell(row,value);
  }
  scroll.append(table); $('metrics').append(scroll);
  const foot = document.createElement('p'); foot.className='footnote'; foot.textContent='¹ Precision, Recall и F1 для рукописного класса. F1 macro — среднее F1 двух классов. «Определено» — доля ответов handwritten или printed.'; $('metrics').append(foot);
  const matrices = document.createElement('div'); matrices.className='matrices';
  for (const [key,m] of entries) {
    const block=document.createElement('div');block.className='matrix';
    const title=document.createElement('h3');title.textContent=`Матрица ошибок · ${key}`;block.append(title);
    const matrix=document.createElement('table'), header=matrix.createTHead().insertRow();
    for (const value of ['Факт ↓ / ответ →','Рукописный','Печатный','Отказ','Формат','Сбой'])cell(header,value,'th');
    for (const label of ['handwritten','printed']) {
      const row=matrix.insertRow();cell(row,names[label]);
      for (const outcome of ['handwritten','printed','unknown','invalid','error'])cell(row,m.confusion[label][outcome],'td',outcome===label?'correct':m.confusion[label][outcome]?'wrong':'');
    }
    block.append(matrix);matrices.append(block);
  }
  $('matrix').append(matrices);
  if (report.report_file && /^[a-z0-9_]+\.json$/i.test(report.report_file)) {
    for (const [label,file] of [['Протокол JSON',report.report_file],['Ответы CSV',report.report_file.replace('.json','.csv')]]) {
      const link=document.createElement('a');link.href='/results/'+file;link.download=file;link.textContent=label;$('report-links').append(link);
    }
  }
  $('prediction-rows').replaceChildren();
  for (const item of report.predictions) {
    const row=document.createElement('tr');
    const imageCell=cell(row,'');const link=document.createElement('a');link.href='/data/'+item.file;link.target='_blank';link.rel='noopener';link.textContent=item.id;imageCell.append(link);
    for(const value of [names[item.truth],item.model,names[item.label],seconds(item.seconds)])cell(row,value);
    $('prediction-rows').append(row);
  }
  $('prediction-details').hidden=false;
}
async function initialize() {
  try {
    const config=await api('/api/config');
    for(const id of ['h01','p01','h03','p03']) {
      const sample=config.samples.find(item=>item.id===id);if(!sample)continue;
      const button=document.createElement('button');button.className='example';button.type='button';
      const img=document.createElement('img');img.src='/data/'+sample.file;img.alt=names[sample.label]+' текст, '+id;
      const caption=document.createElement('span');caption.textContent=names[sample.label];button.append(img,caption);
      button.addEventListener('click',()=>{if(busy)return;selected={sample_id:id};document.querySelectorAll('.example').forEach(x=>x.classList.remove('selected'));button.classList.add('selected');showImage(img.src,`Пример ${id} · набор для разработки`);});
      $('examples').append(button);
    }
    showReport(await api('/api/report'));
    const status=await api('/api/status');if(status.status==='running'){setBusy(true);poll();}
  }catch(exc){error('Не удалось загрузить программу. Проверьте окно запуска и обновите страницу.');}
}
initialize();
