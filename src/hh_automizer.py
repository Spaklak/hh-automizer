import requests
import sys
import os
from bs4 import BeautifulSoup

import time

from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
)

from config import get_config

ERROR = 2
OK = 1
BAD = 0

class ResumeSummary(BaseModel):
    """Модель валидации для подробного анализа резюме."""
    candidate_experience: str = Field(
        description="Подробное описание профессионального опыта кандидата в 2-4 предложениях. Опиши его ключевые роли и обязанности."
    )
    work_experience: List[str] = Field(
        description="Подробное описание каждого места работы с акцентом на достижения, должность и используемые технологии"
    )
    tech_stack: List[str] = Field(
        description="Список ключевых технологий, языков программирования и фреймворков, которыми владеет кандидат."
    )
    key_achievements: str = Field(
        description="Краткое изложение 1-2 самых значимых достижений или проектов, упомянутых в резюме.",
        default="Ключевые достижения не указаны."
    )

class AssessingTheProximity(BaseModel):
    """Модель валидации для оценки близости вакансии и резюме."""
    mark: int = Field(description="Оценка того насколько подходит кандидат к вакансии от 0 до 4 включительно")
    thoughts: str = Field(description="Письменное обоснование выбранной оценки")

class HHAutomizer:
    """Класс для автоматизации откликов на hh.ru"""
    _config = get_config()

    def __init__(self, resume_summary: str=None, temperature: float=0.1, top_p: float=0.95, max_tokens: int = 32000) -> None:
        self.resume_summary = resume_summary
        self.count_page = 0
        self.sum_tokens = 0
        self.model = None
        if self._config['prefered_provider'] == "open_router":
            self.model = ChatOpenAI(
                model=self._config['open_router_model_name'],
                base_url=self._config['open_router_base_url'],
                api_key=self._config['open_router_api'],
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_tokens,
            )
        elif self._config['prefered_provider'] == "giga_chat":
            self.model = GigaChat(
                credentials=self._config['giga_chat_api'],
                model=self._config['giga_chat_model_name'],
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=False,
                profanity=True,
                timeout=600,
                max_tokens=max_tokens,
            )

    @classmethod
    def get_resumes(clf) -> dict:
        """Получает все резюме, которые связаны с пользователем API.
        
        Returns:
            dict: словарь со всеми резюме пользователя
        """
        url = "https://api.hh.ru/resumes/mine"
        headers = {"Authorization": f"Bearer {clf._config['access_token']}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            content = response.json()
        except Exception as e:
            print(f"Ошибка: {e}")
            return None

        if len(content["items"]) > 0:
            for resume in content["items"]:
                print(f"Название: {resume['title']}, id: {resume['id']}")
        else:
            print("У вас нет резюме")
            return None

        return content

    @classmethod
    def get_resume_details(clf) -> dict:
        """Получает детали резюме пользователя API.
        
        Returns:
            dict: словарь, где в ключах указаны различные детали. К примеру, готовность к переезду, локация и т.п.
        """
        url = f"https://api.hh.ru/resumes/{clf._config['resume_id']}"
        headers = {"Authorization": f"Bearer {clf._config['access_token']}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            resume_data = response.json()
            return resume_data
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении резюме: {e}")
            return None

    @classmethod
    def get_similar_vacancies(clf, page: int, per_page: int) -> dict:
        """Получает список с вакансиями с учетом пагинации
            
        Args:
            page (int): страница, с которой берутся вакансии
            per_page (int): количество вакансий, которые будут браться со страницы
        
        Returns:
            dict: словарь с вакансиями 
        """
        url = f"https://api.hh.ru/resumes/{clf._config['resume_id']}/similar_vacancies?page={page}&per_page={per_page}"
        headers = {"Authorization": f"Bearer {clf._config['access_token']}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            content = response.json()
            return content
        except Exception as e:
            print(f"Ошибка: {e}")
            return None

    @staticmethod
    def get_vacancy_details(vacancy_id: str) -> dict:
        """Получает детали вакансии по её id.

        Args:
            vacancy_id (int): id вакансии

        Returns:
            dict: словарь, где в ключах указаны различные детали. К примеру, готовность к переезду, локация и т.п. 
        """
        url = f"https://api.hh.ru/vacancies/{vacancy_id}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            vacancy = response.json()
            return vacancy
        except Exception as e:
            print(f"Ошибка для вакансии: {vacancy_id}: {e}")
            return None
    
    @staticmethod
    def resume_parser(resume: dict) -> str:
        """Метод парсинга резюме по ключам, которые передаются hh.ru.

        Args:
            resume (dict): словарь, где в ключах указаны различные детали резюме, которые предоставляются hh.ru

        Returns:
            str: строка с опытом работы, навыками и указанным набором навыков
        """
        experience_descriptions = []
        for exp in resume.get("experience", []):
            company = exp.get("company", "Не указано")
            position = exp.get("position", "Не указано")
            description = exp.get("description", "Нет описания")
            experience_descriptions.append(
                f"Компания: {company}\nДолжность: {position}\nОписание: {description}\n"
            )
        experience_info = (
            "\n".join(experience_descriptions)
            if experience_descriptions
            else "Опыт работы не указан."
        )
        skills = resume.get("skills", "Навыки не указаны.")
        skill_set = resume.get("skill_set", [])
        skill_set_info = (
            ", ".join(skill_set) if skill_set else "Набор навыков не указан."
        )
        resume_description = (
            f"Опыт работы:\n{experience_info}\n\n"
            f"Навыки:\n{skills}\n\n"
            f"Набор навыков:\n{skill_set_info}"
        )
        return resume_description

    def set_resume_summary(self) -> None:
        """Устанавливает краткое описание резюме кандидата, выделяя главные особенности из его опыта."""
        if self.resume_summary is None:
            resume = self.get_resume_details()
            resume_description = self.resume_parser(resume)
            prompt = PromptTemplate.from_template(
                template="""
                # РОЛЬ
                Ты — опытный IT-рекрутер. Твоя задача — провести детальный анализ резюме и составить структурированную выжимку для нанимающего менеджера.

                # КОНТЕКСТ
                Мне нужна подробная сводка из резюме, которая поможет оценить профессиональный уровень кандидата, его технические навыки и реальные достижения. Вся информация должна быть извлечена строго из предоставленного текста. Не додумывай и не делай предположений.

                **Резюме для анализа:**
                {resume_description}
                ---

                # ИНСТРУКЦИЯ
                Проанализируй текст резюме ниже и представь результат в формате Markdown, используя следующую структуру:

                **АНАЛИЗ РЕЗЮМЕ**

                **1. Опыт работы (с акцентом на достижения):**
                -   Для каждого места работы укажи:
                    -   **Компания:** Название компании
                    -   **Должность:** Название должности
                    -   **Период:** Даты работы
                    -   **Ключевые достижения и обязанности:** Выдели 2-3 самых важных пункта. Сфокусируйся на измеримых результатах (например, "увеличил производительность на 20%", "сократил время ответа сервера на 300 мс", "разработал и внедрил систему X").

                **2. Технические навыки (Hard Skills):**
                -   **Языки программирования:**
                -   **Фреймворки и библиотеки:**
                -   **Базы данных:**
                -   **Инструменты и технологии:** [CI/CD, Docker, Kubernetes, Облачные сервисы (AWS, GCP) и т.д.]
                -   **Методологии:** [Agile, Scrum, Kanban]
                """
            )
            structured_llm = self.model.with_structured_output(ResumeSummary)
            chain = prompt | structured_llm
            try:
                response: ResumeSummary = chain.invoke({"resume_description": resume_description})
                summary_text = (
                    f"Краткое содержание резюме:\n"
                    f"Опыт кандидата: {response.candidate_experience}\n"
                    f"Места работы:\n"
                    f"{'\n'.join(response.work_experience)}\n"
                    f"Технологический стек: {', '.join(response.tech_stack)}\n"
                    f"Ключевые достижения: {response.key_achievements}\n"
                )
                self.resume_summary = summary_text
                with open("resume_summary.txt", "w", encoding="utf-8") as file:
                    file.write(summary_text)
            
            except Exception as e:
                print(f"Не удалось сгенерировать краткое содержание резюме: {e}")
                self.resume_summary = None

    def get_mark_resume_vac(self, vac_describe: str) -> int:
        """Метод для оценки подходит ли резюме по описанию вакансии.

        Args:
            vac_describe (str): текстовое описание вакансии
        
        Returns:
            int: ERROR - ошибка, BAD - не подходит, OK - подходит
        """
        if self.resume_summary is None:
            raise ValueError("Резюме отсутвует. Невозможно оценить его с вакансией")
        
        prompt = PromptTemplate.from_template("""
        # РОЛЬ
        Ты - опытный IT-рекрутер с многолетним стажем. Твоя главная задача - оценить насколько кандидат по его резюме подходит к выбранной вакансии.

        # ИНСТРУКЦИИ ПО ОЦЕНКЕ
        Ты проводишь оценку от 0 до 4х баллов включительно. Вот критерии, которыми ты должен руководиться:
        - 0 баллов. Вакансия не имеет отношения к сфере деятельности кандидата. К примеру, кандидат работает в IT, а вакансия продавца техники
        - 1 балл. Полное несоответствие вакансии и резюме. Нет пересечений по стеку, либо это очень далекие пересечения, ввиду которых кандидату будет довольно сложно пройти собеседование.
        - 2 балла. Кандидат едва подходит по вакансии. Да, шансы пройти собеседования есть, но они слишком малы. Нужно многое изучить, чтобы попасть на эту вакансию
        - 3 балла. Кандидат имеет неплохие шансы. Довольно высокое пересечение стека и навыков между резюме и вакансией. Есть что изучить, но не так много.
        - 4 балла. Идеальное совпадение. Стек и навыки кандидата с излишками покрывают требования вакансии, либо полностью соответствуют им. Кандидат имеет высокие шансы пройти собеседование и получить оффер.

        # КОНТЕКСТ
        Вакансия, которая предлагается кандидату:
        {vacancy}
        ====================
        Резюме кандидата:
        {resume}
        """)
        structured_llm = self.model.with_structured_output(AssessingTheProximity)
        chain = prompt | structured_llm
        try:
            response: AssessingTheProximity = chain.invoke({"vacancy": vac_describe, "resume": self.resume_summary})
            print(f"Оценка LLM: {response.mark}\nОбоснование оценки: {response.thoughts}")
            if response.mark >= self._config["criterion"]["mark"]:
                return OK
            else:
                BAD
        except Exception as e:
            print(f"Ошибка во время оценки сходимости резюме и вакансии: {e}")
            return ERROR

    def create_cover_letter(self, vacancy_name: str, vacancy_description: str) -> str:
        """Создает сопроводиельное письмо.
        
        Args:
            vac_name (str): название вакансии
            vac_describe (str): описание вакансии
        
        Returns:
            str: готовое сопроводительное письмо
        """
        prompt = PromptTemplate.from_template(template="""
        # РОЛЬ
        Ты - опытный IT-рекрутер с многолетним стажем. Твоя главная задача - составить хорошее сопроводительное письмо, исходя из резюме кандидата и выбранной вакансии.
        Однако это не письмо на почту, а сообщение в чат в приложение по типу Linkedin, тем не менее оно должно быть официальным, но не нужно формальностей по типу "от кого" и "кому"

        # Инструкции
        0. Обязательно представься от имени кандидата и укажи на какую позицию хочет кандидат
        1. Учти, что обязательно нужно описать, какие технологии помогут кандидату на его новом месте, исходя из требований вакансии
        2. Отметь опыт, который есть у кандидата для данной вакансии. Именно практический опыт. К примеру, разработал модель машинного обучения, которая увеличила точность прогнозирования на 25%. Опыт ты должен брать из резюме кандидата
        3. Отметь, какой стимул имеет кандидат, чтобы начать работу именно в этой компании. Тут важно показать, что кандидат изучил продукты кампании, ценности и проекты
        4. В конце поблагодари за внимание и вырази готовность кандидата к собеседованию, передав нужные контакты
        5. Письмо не должно быть длинным. Важно, чтобы его можно было прочесть за 10-20 секунд
        6. В конце нужно указать контакты кандидаты для обратной связи.
        7. Не нужно указывать точный технологический стек кандидата. Ты можешь упоминать общие технологии по типу: ML, RAG, MCP и т.д., но средства реализации по типу Langchain, XGBoost указывать не стоит

        # Контекст
        Текст вакансии:
        =========================
        Название вакансии: {vacancy_name}
        {vacancy_description}

        Текст резюме кандидата:
        =========================
        {resume_description}

        Контакты кандидата:
        =========================
        - Имя: {name}
        - Телеграм: @{telegram}
        - Почта: {email}

        # Структура ответа
        В качестве ответа выдай только составленное сопроводительное письмо, учтя все инструкции, которые были переданы
        """
        )
        chain = prompt | self.model

        try:
            response = chain.invoke({
                "vacancy_description": vacancy_description,
                "vacancy_name": vacancy_name,
                "resume_description": self.resume_summary,
                "name": self._config['candidate']['name'],
                "telegram": self._config['candidate']['telegram'],
                "email": self._config['candidate']['email']
            })
            self.sum_tokens += response.usage_metadata['total_tokens']
            return response.content
        except Exception as e:
            print(f"Ошибка во время формирования сопроводительного письма. Текст ошибки: {e}")
            return None

    def apply_to_vacancy(self, vacancy_id: str, cover_letter: str) -> None:
        """Отправляет отклик на вакансию.
        
        Args:
            vacancy_id (str): id вакансии
            cover_letter (str): сопроводительное письмо
        """
        url = "https://api.hh.ru/negotiations"
        headers = {
            "Authorization": f"Bearer {self._config['access_token']}",
        }
        data = {
            "vacancy_id": vacancy_id,
            "resume_id": self._config["resume_id"],
            "message": cover_letter,
        }
        try:
            response = requests.post(url, headers=headers, data=data)
            if response.status_code == 201:
                print(f"Отлик успешно отправлен. ID вакансии: {vacancy_id}")
            elif response.status_code == 303:
                redirect_url = response.headers.get("Location")
                print(f"Требуется коректный url: {redirect_url}")
            else:
                print(
                    f"Ошибка отправки отлика на вакансию {vacancy_id}: {response.status_code} {response.text}"
                )
        except requests.exceptions.RequestException as e:
            print(f"Ошибка отправки отлика на вакансию {vacancy_id}: {e}")

    def make_responses(self) -> None:
        """Главный метод, который инициализирует цикл откликов"""
        if self.resume_summary is None:
            raise ValueError("Резюме отсутвует. Невозможно начать делать отклики")

        done_responses = 0
        reviewed_vacancies = 0
        while done_responses < self._config["criterion"]["number_of_responses"]:
            time.sleep(0.5)
            similar_vacancies = self.get_similar_vacancies(
                page=self.count_page, per_page=100
            )
            self.count_page += 1
            if len(similar_vacancies) > 0:
                for vacancy in similar_vacancies["items"]:
                    time.sleep(0.5) # hardcoded. Введено из-за ограничения API hh.ru на количество запросов в секунду
                    vacancy_id = vacancy["id"]
                    vacancy_details = self.get_vacancy_details(vacancy_id)
                    if vacancy_details:
                        vac_name = vacancy_details.get("name", "Не указано")
                        vac_description = vacancy_details.get(
                            "description", "Не указано"
                        )
                        vac_experience = vacancy_details.get("experience", {}).get(
                            "name", "Не указано"
                        )
                        vac_company = vacancy_details.get("employer", {}).get(
                            "name", "Не указано"
                        )
                        soup = BeautifulSoup(vac_description[:], "html.parser")
                        vac_description_clear = soup.get_text()
                        reviewed_vacancies += 1
                        print(f"Название компании: {vac_company}")
                        if vac_experience in self._config["criterion"]["experience"]:
                            mark = self.get_mark_resume_vac(vac_description_clear)
                            if mark == OK:
                                print(f"Название вакансии: {vac_name}")
                                print(f"Требуемый опыт работы: {vac_experience}")

                                cover_letter = self.create_cover_letter(
                                    vac_name, vac_description_clear
                                )
                                time.sleep(0.5)
                                if isinstance(cover_letter, str):
                                    print(f"Сопроводительное письмо:\n{cover_letter}")
                                    self.apply_to_vacancy(vacancy_id, cover_letter)
                                    done_responses += 1
                                    print("-" * 40)
                                else:
                                    print(f"Не удалось составить сопроводительное письмо. Вакансия {vac_name}. ID вакансии: {vacancy_id}")
                                    print("-" * 40)
                            elif mark == BAD:
                                print(
                                    f"Вакансия {vac_name} не подходит по требованиям. ID вакансии: {vacancy_id}"
                                )
                                print("-" * 40)
                            else:
                                print(f"Ошибка при оценке вакансии {vac_name}")
                                print("-" * 40)
                        else:
                            print(
                                f"Вакансия {vac_name} не подходит по опыту работы. Требуемый опыт работы: {vac_experience}. ID вакансии: {vacancy_id}"
                            )
                            print("-" * 40)

                    if (
                        done_responses
                        == self._config["criterion"]["number_of_responses"]
                    ):
                        break
            else:
                break
        print("=" * 40)
        print(f"\nВсего использовано токенов: {self.sum_tokens}\n")
        print(f"Обработано вакансий: {reviewed_vacancies}\n")
        print(f"Откликов сделано: {done_responses}\n")
        print("=" * 40)
