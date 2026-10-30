
prices = [500, 1200, 800, 2500, 300]
def apply_discount(price, discount=10):
    return price - (price * discount / 100)
def total(*args):
    return sum(args)
def describe(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")
print("apply_discount(1000) ->", apply_discount(1000))
print("apply_discount(1000, 25) ->", apply_discount(1000, 25))
print("total(5, 10, 15) ->", total(5, 10, 15))
print("\nDescription:")
describe(name="Laptop", brand="Dell", price=50000)
above_1000 = [price for price in prices if price > 1000]
print("\nAbove 1000:", above_1000)
with_gst = list(map(lambda price: round(price * 1.18, 2), prices))
print("With GST:", with_gst)
below_1000 = list(filter(lambda price: price < 1000, prices))
print("Below 1000:", below_1000)