# STEP 1 : import modules
import easyocr

# STEP 2 : create inference object
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# STEP 3 : load data
data = '3695ca1384edb56557fcf57c5392b843.jpg'

# STEP 4 : inference
result = reader.readtext(data)
print(result)

# STEP 5 : post processing
