from src.preprocessing.wind_processor import WINDProcessor

def main():

    processor=WINDProcessor()

    processor.process_all()

if __name__=="__main__":

    main()