class InsufficientFundsError(Exception):
    """Custom exception for insufficient balance"""
    pass


class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{self.owner} deposited {amount} -> balance {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount
        print(f"{self.owner} withdrew {amount} -> balance {self.balance}")

    def show(self):
        print(f"Owner: {self.owner}, Balance: {self.balance}")


class SavingsAccount(Account):
    def __init__(self, owner, balance=0):
        super().__init__(owner, balance)   # Reuse parent constructor
        self.min_balance = 500

    def withdraw(self, amount):
        if self.balance - amount < self.min_balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount
        print(f"{self.owner} withdrew {amount} -> balance {self.balance}")


# Test
priya = SavingsAccount("Priya", 1500)

try:
    priya.withdraw(1200)   # Not allowed: leaves only 300
except InsufficientFundsError as e:
    print("Error:", e)

try:
    priya.withdraw(800)    # Allowed: leaves 700
except InsufficientFundsError as e:
    print("Error:", e)

