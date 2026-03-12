from custom_classes import MediaItem, Book, DVD, LibraryCollection


book_1 = MediaItem("Kumarbaz","Dostoyevski",True)
movie_1 = MediaItem("Ahlat Ağacı","Nuri Bilge Ceylan",False)

print(book_1.checkout()) # Book 1 checked out
print(book_1.checkout()) # Book 1 checked out again but not available
print(movie_1.checkout()) # Movie 1 checked out but not available

# --- Part 2 ---
book_2 = Book("Korluk", "Jose Saramago", True, 265) # New book and movie added with subclasses
movie_2 = DVD("Martian", "Unknown", True, 173) 

print(book_2) # View the info about book_2
print(book_2.checkout()) # Book_2 checked out
print(movie_2) # View the info about movie_2
print(movie_2.checkout()) # Movie_2 checked out

# --- Part 3 ---
library = LibraryCollection() # Library created

library.add_item(book_1) # New items added to the library
library.add_item(book_2)
library.add_item(movie_1)
library.add_item(movie_2)

print(library.list_available()) # This gave us an empty list because we already checked out items

book_1.return_item() # Returned items
book_2.return_item()
movie_2.return_item()

print(library.list_available()) # Now we can view the available items in our library
