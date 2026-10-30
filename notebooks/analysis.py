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


# Test
account = Account("Arun", 1000)

try:
    account.deposit(500)
    account.withdraw(2000)

except InsufficientFundsError as e:
    print("Error:", e)

finally:
    print("Transaction finished")