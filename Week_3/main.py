from custom_classes import MediaItem, Book, DVD, LibraryCollection


book_1 = MediaItem("Kumarbaz","Dostoyevski",True)
movie_1 = MediaItem("Ahlat Ağacı","Nuri Bilge Ceylan",False)

print(book_1.checkout())
print(book_1.checkout())
print(movie_1.checkout())

# --- Part 2 ---
book_2 = Book("Korluk", "Jose Saramago", True, 265)
movie_2 = DVD("Martian", "Unknown", True, 173)

print(book_2)
print(book_2.checkout())
print(movie_2)
print(movie_2.checkout())

# --- Part 3 ---
library = LibraryCollection()

library.add_item(book_1)
library.add_item(book_2)
library.add_item(movie_1)
library.add_item(movie_2)

print(library.list_available())

book_1.return_item()
book_2.return_item()
movie_2.return_item()

print(library.list_available())
