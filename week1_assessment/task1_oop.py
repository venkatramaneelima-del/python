import csv


# Custom Exception
class InvalidSalaryError(Exception):
    pass


# Bonus Function
def bonus(salary, rate=0.1, *extras, **options):
    total = salary + (salary * rate)

    for extra in extras:
        total += extra

    if "cap" in options:
        total = min(total, options["cap"])

    return total


# Employee Class
class Employee:
    def __init__(self, name, salary):
        if salary < 0:
            raise InvalidSalaryError("Salary cannot be negative")

        self.name = name
        self.salary = salary

    def annual_salary(self):
        return self.salary * 12

    def get_details(self):
        return f"Employee: {self.name}, Salary: {self.salary}"


# Manager Class
class Manager(Employee):
    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)
        self.team_size = team_size

    def get_details(self):
        return (f"Manager: {self.name}, "
                f"Salary: {self.salary}, "
                f"Team Size: {self.team_size}")


# Read CSV File
employees = []

with open("week1_assessment/employees.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        row["salary"] = int(row["salary"])
        employees.append(row)


# List Comprehension
high_salary = [
    emp["name"]
    for emp in employees
    if emp["salary"] > 50000
]

print("Salary > 50000:")
print(high_salary)


# Map + Lambda (10% hike)
raised_salary = list(
    map(lambda emp: emp["salary"] * 1.10, employees)
)

print("\nAfter 10% Hike:")
print(raised_salary)


# Filter Chennai Employees
chennai_emp = list(
    filter(lambda emp: emp["city"] == "Chennai", employees)
)

print("\nChennai Employees:")
for emp in chennai_emp:
    print(emp["name"])


# Bonus Function Demo
print("\nBonus Salary:")
print(bonus(50000, 0.1, 2000, 1000, cap=60000))


# OOP Demo
manager = Manager("Rahul", 60000, 5)
print("\nManager Details:")
print(manager.get_details())
print("Annual Salary:", manager.annual_salary())


# Exception Handling Demo
try:
    e = Employee("Test", -1000)
except InvalidSalaryError as err:
    print("\nError:", err)
finally:
    print("Program Completed.")