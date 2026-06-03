from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch
from dotenv import load_dotenv

load_dotenv()   

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range:'
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
    }}
"""


#Update your restaurant_data_structure_prompt_generation
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a helpful assistant who extracts structured information about restaurants from unstructured text.
    """
    
    base_user_prompt = f"""
    Task:
    Given a restaurant description, extract the following information and present it in a structured JSON format:
    - name: The name of the restaurant
    - location: The location of the restaurant
    - type: The type of the restaurant
    - food_style: The style of food served
    - rating: The rating of the restaurant
    - price_range: The price range of the restaurant
    - signatures: A list of signature dishes
    - vibe: The vibe of the restaurant
    - environment: The environment of the restaurant
    - shortcomings: A list of shortcomings

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    
    """
    return base_system_msg, base_user_prompt

# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt, params=None):
    #system_msg: the system message given to the LLM
    #prompt_txt: the user prompt
    
    model_id = "ibm/granite-4-h-small"
    project_id=os.getenv('WATSONX_PROJECT_ID')
    credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com"
                              , api_key =os.getenv('WATSONX_API_KEY'))

    ### 1.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        project_id=project_id,
        credentials=credentials
    )

    ### 1.2: Define the messages
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    ### 1.3: Get the final response output and return it
    response = model.chat(messages=messages)
    final_output = response["choices"][0]["message"]["content"]
    return final_output

def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """
    You are a helpful assistant who corrects the JSON output based on the error message from the validation.
    """
    auto_repair_prompt = f"""
    The following JSON output is invalid:
    {response}

    Error message:
    {error_message}

    Please correct the JSON output according to the error message.
    """
    return auto_repair_system_msg, auto_repair_prompt


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


def new_data_entry_process(paragraph, itemId):
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    candidate_response = llm_model(system_msg, user_prompt)

    max_repair_rounds = 2
    repair_attempt = 0
    restaurant_data = None
    while repair_attempt <= max_repair_rounds:
        try:
            restaurant_data = Restaurant.model_validate_json(candidate_response)
            break
        except ValidationError as e:
            if repair_attempt == max_repair_rounds:
                break
            error_message = e.json()
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(
                candidate_response, error_message
            )
            candidate_response = llm_model(auto_repair_system_msg, auto_repair_prompt)
            repair_attempt += 1

    if restaurant_data is None:
        return None

    entry = restaurant_data.model_dump()
    entry['itemId'] = itemId
    return entry


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, file_path, backup_path):
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    print(f"\n--- Record #{index} ---")
    for key, value in res.items():
        print(f"{key}: {value}")


def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, record in enumerate(data):
                print(f"{i}: {record.get('name', 'N/A')}")

        elif choice == '2':
            try:
                index = int(input("Enter record index: "))
            except ValueError:
                print("invalid index.")
                continue
            if 0 <= index < len(data):
                show_restaurant_card(data[index], index)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter the new restaurant description: ")
                new_entry = new_data_entry_process(paragraph, itemId)
                if new_entry is None:
                    print("❌ Failed to process restaurant description.")
                else:
                    data.append(new_entry)
                    save_data(data, file_path, backup_path)
                    print("✅ Restaurant added.")

            elif choice == '4':  # EDIT DATA
                try:
                    index = int(input("Enter record index: "))
                except ValueError:
                    print("invalid index.")
                    continue
                if not (0 <= index < len(data)):
                    print("invalid index.")
                    continue
                record = data[index]
                for key in record:
                    new_val = input(f"New value for '{key}' (Enter to skip): ")
                    if new_val.strip() == '':
                        continue
                    current = record[key]
                    if isinstance(current, list):
                        data[index][key] = [item.strip() for item in new_val.split(',')]
                    elif isinstance(current, int):
                        data[index][key] = int(new_val)
                    elif isinstance(current, float):
                        data[index][key] = float(new_val)
                    else:
                        data[index][key] = new_val
                save_data(data, file_path, backup_path)
                print("✅ Record updated.")

            elif choice == '5':  # DELETE DATA
                try:
                    index = int(input("Enter record index: "))
                except ValueError:
                    print("invalid index.")
                    continue
                if 0 <= index < len(data):
                    data.pop(index)
                    save_data(data, file_path, backup_path)
                    print("✅ Restaurant deleted.")
                else:
                    print("invalid index.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")

class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    # unittest.main()  # Unit Test
    manage_restaurants(FILEPATH, BACKUP_PATH)  # Actual UI Call
