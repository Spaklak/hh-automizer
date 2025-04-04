import requests
import sys
import os
from bs4 import BeautifulSoup

import time

from langchain_community.chat_models import GigaChat

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
)

from config import get_config

ERROR = 2
OK = 1
BAD = 0


class HHAutomizer:
    _config = get_config()

    def __init__(self, resume_summary=None) -> None:
        self.resume_summary = resume_summary
        self.count_page = 0
        self.sum_tokens = 0

    @classmethod
    def get_resumes(clf) -> dict:
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
        url = f"https://api.hh.ru/vacancies/{vacancy_id}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            vacancy = response.json()
            return vacancy
        except Exception as e:
            print(f"Ошибка для вакансии: {vacancy_id}: {e}")
            return None

    def set_resume_summary(self) -> None:
        if self.resume_summary is None:
            resume = self.get_resume_details()
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
            user_prompt = """Выдели главные особеннности, которые помогли бы помочь при при подборе вакансии. В ответе дай только особенности, которые ты выделил в тексте.
            Текст:
            ====================
            {resume_description}
            ====================
            """.format(
                resume_description=resume_description
            )
            system_prompt = "Ты — Олег, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только строго по контексту."

            chat = GigaChat(
                credentials=self._config["giga_chat_api"],
                scope="GIGACHAT_API_PERS",
                verify_ssl_certs=False,
                profanity=True,
                timeout=600,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = chat.invoke(messages)

            self.resume_summary = response.content.replace("\n", "")
            with open("resume_summary.txt", "w", encoding="utf-8") as file:
                file.write(response.content)

    def get_mark_resume_vac(self, vac_describe: str) -> int:
        if self.resume_summary is None:
            raise ValueError("Резюме отсутвует. Невозможно оценить его с вакансией")

        user_prompt = """Тебе нужно определить по 10ти бальной шкале насколько вакансия подкодит по резюме. Учти, что пользователь хочет быть DS специалистом. 
        Ответ выдай в формате "Размышления: (твои мысли) - Оценка: (от 1 до 10). Только одно число".
        Вакансия:
        ====================
        {vacancy}
        ====================
        Резюме:
        ====================
        {resume}
        ====================
        """.format(
            vacancy=vac_describe, resume=self.resume_summary
        )
        system_prompt = "Ты — Олег, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только строго по контексту."

        chat = GigaChat(
            credentials=self._config["giga_chat_api"],
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,
            profanity=True,
            timeout=600,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = chat.invoke(messages)
        self.sum_tokens += response.response_metadata["token_usage"].total_tokens
        try:
            mark = int(response.content.split("Оценка: ")[1])
            print(f"Оценка LLM: {mark}")
            if mark >= self._config["criterion"]["mark"]:
                return OK
            else:
                return BAD
        except:
            return ERROR

    def create_cover_letter(self, vac_name, vac_describe) -> str:
        chat = GigaChat(
            credentials=self._config["giga_chat_api"],
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False,
            profanity=True,
            timeout=600,
        )
        user_prompt = """
        Привет, мне нужно, чтобы сейчас мне помогал составлять сопроводительные письма в разные компании. Я хочу работать DS специалистом. 

        В следующем тексте есть пропущенные поля, обозначенные _ и _. Также в этих подчеркиваниях есть инструкция что нужно вставить. Твоя задача написать мне сопроводительное письмо с заполненными полями. В ответе нужно выдать только само письмо
        Шаблон:
        Добрый день,

        Меня зовут Иван, я работаю в сфере Data Science уже 1.5 года и хотел бы предложить свою кандидатуру на вашу позицию {vacancy_name}.

        Последнее время я активно занимаюсь применением LLM-моделей в IT-решениях: создаю ботов, которые упрощают жизнь пользователей, настраиваю RAG-пайплайны и разрабатываю AI-агентов.

        _тут 2-3 предложения почему я хочу работать в этой компании_

        Буду рад обсудить детали вакансии и ответить на ваши вопросы. Со мной можно связаться в Telegram @Ivan или по почте ivan.ivanov@vanya.ru.

        С уважением,
        Иван
        Текст вакансии:
        =========================
        {vacancy_descr}
        """.format(
            vacancy_name=vac_name, vacancy_descr=vac_describe
        )

        system_prompt = "Ты — Олег, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только строго по контексту."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = chat.invoke(messages)
        self.sum_tokens += response.response_metadata["token_usage"].total_tokens
        return response.content

    def apply_to_vacancy(self, vacancy_id: str, cover_letter: str) -> None:
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
                    time.sleep(0.5)
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
                                self.apply_to_vacancy(vacancy_id, cover_letter)
                                done_responses += 1
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
