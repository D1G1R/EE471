class MediaItem(object):
    def __init__(self, title, author, is_available):
        self.title = title
        self.author = author
        self.is_available = is_available

    def checkout(self):
        if self.is_available == True: # check if is.available is True or False
            self.is_available = False # if True than set it to False
            return("Succesful checkout.")
        else:
            return("Already out, sorry.") 
        
    def return_item(self):
        self.is_available = True

    def __str__(self):
        if self.is_available == True: # A temporary variable 
            temp = "Available"
        else:
            temp = "Checked Out"
        return str(self.title) + " by " + str(self.author) + " [" + temp + "]"
    
# --- Part 2 --- 
class Book(MediaItem):
    def __init__(self, title, author, isavailable, page_count):
        super().__init__(title, author, isavailable) # super used to gather features from MediaItem
        self.page_count = page_count

    def __str__(self):
        return super().__str__() + ", Page: " + str(self.page_count)
    
class DVD(MediaItem):
    def __init__(self, title, author, isavailable, duration):
        super().__init__(title, author, isavailable)
        self.duration = duration

    def __str__(self):
        return super().__str__() + ", Duration: " + str(self.duration)

    def checkout(self):
        return super().checkout() + " Handle with care: Do not scratch the disc."
    
#Question: What is the use of super() here? What happens if we change the logic MediaItem.Checkout() later?
#Ans: Super() calls the parent class's method so we can reuse and extend its logic without rewriting code. 
# If we update MediaItem.checkout(), the changes automatically apply to all subclasses. This ensures consistency and makes the code much easier to maintain.
# --- Part 3 ---
class LibraryCollection(object):
    def __init__(self):
        self.items = [] # An empty list created so we can add items later

    def add_item(self, new_item):
        self.items.append(new_item)

    def list_available(self):
        available_titles = []

        for item in self.items:
            if item.is_available:
                available_titles.append(item.title)

        return available_titles
        # "return [item.title for item in self.items if item.is_available]" this is the shorter way



    