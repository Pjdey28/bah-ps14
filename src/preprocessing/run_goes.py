from src.preprocessing.goes_processor import GOESProcessor

def main():

    processor=GOESProcessor()

    processor.process_all()

if __name__=="__main__":

    main()