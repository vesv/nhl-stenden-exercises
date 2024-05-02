#exercise 1
def id_to_fruit(fruit_id: int, fruits: list[str]) -> str:
    if fruit_id < len(fruits):
        return fruits[fruit_id]
    else:
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

name1 = id_to_fruit(1, ["apple", "orange", "melon", "kiwi", "strawberry"])
name3 = id_to_fruit(3, ["apple", "orange", "melon", "kiwi", "strawberry"])
name4 = id_to_fruit(4, ["apple", "orange", "melon", "kiwi", "strawberry"])
print(name3)

#There were 2 reasons for the error:
#firstly, in the definition 'set' was written with a capital letter which is invalid
#secondly, because sets in python are unordered, the code was returning random fruits
#this means that when you iterate over the set, the order of the elements is not guaranteed
#to fix this, the set should be converted to a list