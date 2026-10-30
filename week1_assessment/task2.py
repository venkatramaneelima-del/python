import csv
import json
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# 1. Read employees.csv with csv.DictReader, write employees.json and high_earners.csv
rows_read = 0
high_earners_count = 0
all_employees = []
high_earners_list = []

try:
    with open("employees.csv", mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        
        for row in reader:
            rows_read += 1
            # Convert salary to float/int for comparison
            salary = float(row["salary"]) if "salary" in row else 0.0
            
            all_employees.append(row)
            if salary > 50000:
                high_earners_count += 1
                high_earners_list.append(row)

    # Write all employees to employees.json
    with open("employees.json", mode="w", encoding="utf-8") as json_file:
        json.dump(all_employees, json_file, indent=4)

    # Write high earners to high_earners.csv
    with open("high_earners.csv", mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(high_earners_list)

    print(f"Rows read: {rows_read} | High earners: {high_earners_count}")

except FileNotFoundError:
    print("Error: employees.csv not found.")

# 2. API Requests, Timeouts, and Error Handling
url = "https://jsonplaceholder.typicode.com/users"

try:
    # Fetch first 3 users
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    users = response.json()
    
    for user in users[:3]:
        print(f"{user.get('id')} - {user.get('name')} - {user.get('email')}")

    # Test 404 error handling on /users/9999
    error_url = "https://jsonplaceholder.typicode.com/users/9999"
    error_response = requests.get(error_url, timeout=5)
    if error_response.status_code == 404:
        print(f"/users/9999 -> 404 Not found")

except requests.exceptions.Timeout:
    print("The request timed out.")
except requests.exceptions.ConnectionError:
    print("A connection error occurred.")
except requests.exceptions.HTTPError as err:
    if err.response.status_code == 404:
        print("/users/9999 -> 404 Not found")
    else:
        print(f"HTTP error occurred: {err}")

# 3. Parse a nested JSON field with .get() and save the result to users.json
try:
    response = requests.get(url, timeout=5)
    users_data = response.json()
    
    parsed_users = []
    for user in users_data:
        # Using .get() for safe parsing of nested fields (e.g., address -> city)
        address = user.get("address", {})
        city = address.get("city")
        
        company = user.get("company", {})
        company_name = company.get("name")
        
        parsed_users.append({
            "id": user.get("id"),
            "name": user.get("name"),
            "city": city,
            "company": company_name
        })

    with open("users.json", mode="w", encoding="utf-8") as f:
        json.dump(parsed_users, f, indent=4)

except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")