class MediaItem(object):
    def __init__(self, title, author, is_available):
        self.title = title
        self.author = author
        self.is_available = is_available

    def checkout(self):
        if self.is_available == True:
            self.is_available = False
            return("Succesful checkout.")
        else:
            return("Already out, sorry.")
        
    def return_item(self):
        self.is_available = True

    def __str__(self):
        if self.is_available == True:
            temp = "Available"
        else:
            temp = "Checked Out"
        return str(self.title) + " by " + str(self.author) + " [" + temp + "]"
    
class Book(MediaItem):
    def __init__(self, title, author, isavailable, page_count):
        super().__init__(title, author, isavailable)
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
    
class LibraryCollection(object):
    def __init__(self):
        self.items = []

    def add_item(self, new_item):
        self.items.append(new_item)

    def list_available(self):
        available_titles = []

        for item in self.items:
            if item.is_available:
                available_titles.append(item.title)

        return available_titles
        #return [item.title for item in self.items if item.is_available]



    