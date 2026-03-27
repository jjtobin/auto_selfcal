from casatools import image as iatool

class ImageReader:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.ia = iatool()
        self.ia.open(self.filename)

        return self.ia
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ia:
            self.ia.close()