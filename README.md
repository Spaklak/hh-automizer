# Автоматизатор откликов на hh.ru с использованием GigaChat

## Оглавление
1. [Описание](#описание)
2. [Установка](#установка)
3. [Настройка](#настройка)
4. [Использование](#использование)
5. [Планы](#планы)

## Описание

**HHAutomizer** — это интеллектуальный автоматизатор для платформы hh.ru, созданный для упрощения поиска работы и массовой отправки персонализированных откликов. Проект сочетает возможности API hh.ru и мощь языковой модели GigaChat, чтобы анализировать вакансии, оценивать их соответствие вашему резюме и генерировать уникальные сопроводительные письма.

Используемые технологии:
- **Интеграция с hh.ru API** для работы с резюме и вакансиями.
- **GigaChat** (через LangChain) для NLP-задач.
- **BeautifulSoup** для очистки HTML-описаний вакансий.
- Конфигурационная система с настройкой критериев поиска.

## Установка

Склонируйте репозиторий и установите необходимые зависимости.

```
git clone https://github.com/Spaklak/hh_automizer.git
cd hh_automizer
pip install -r requirements.txt
```

## Настройка

Перед использованием вам нужно сделать следующее:
1. Получить доступ к API.
Для этого нужно перейти по ссылке: https://dev.hh.ru/. Зарегистрировать аккаунт, подать заявку на приложение и дождаться одобрения заявки.
2. Получить ключ для GigaChat.
Для этого нужно перейти по ссылке: https://developers.sber.ru/, зарегистрировать личный кабинет, создать пространство и получить ключ. Также, если вы пользуетесь впервые, то можете получить бесплатно около миллиона токенов.
3. Полученные ключи и критерии для поиска нужно внести в [config.yaml](config/config.yaml).

## Использование

Рекомендую следовать следующему порядку, чтобы не возникло никаких проблем. Также в каждом ноутбуке подробно расписаны все шаги и вариации.

1. Нужно заполнить недостающие поля в `config.yaml`. Для этого нам нужно 2 ноутбука
- Переходим в [get_all_codes.ipynb](notebooks/get_all_codes.ipynb) и получаем **access_token**. Инструкция есть в ноутбуке (Делать этот шаг нужно каждый час, поскольку access_token сбрасывается и его нужно обновлять. Но ограничение у сайта вроде 100 откликов в день, поэтому данную операцию вы будете проделывать раз в день т.к. 100 отлкиков вы наберете быстрее, чем за час).
- Переходим в [get_resume_id.ipynb](notebooks/get_resume_id.ipynb) и оттуда получаем список наших резюме с их названием и ID. Выбираем один и вносим его в конфиг файл.
2. Переходим в [hh_automizer.py](src/hh_automizer.py) и на 199 строке пишем свое сопроводительное письмо. Шаблон уже указан.
3. Переходим в [make_responses.ipynb](notebooks/make_responses.ipynb) и, следуя инструкции, получаем нужное нам количество откликов.

## Планы

В планах добавить большее количество критериев для взаимодействия с внутренними фильтрами hh.ru