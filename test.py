import time

import matplotlib.pyplot as plt

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel


def main():
    model = BestModelEverLOL.load()
    lm = LanguageModel.load()

    # Список отзывов для обработки
    reviews = [
        "The location is in a strip mall, but this place is a diamond in the rough.  The food was some of the best italian food I have had anywhere.  I live in Kansas City, which has a pretty large and long standing italian populaiton.  The food is a good as any of the old italian places in KC.  Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "I live in Kansas City, which has a pretty large and long standing italian population. Everyone in my group was amazed at the quality of the food for the price.  I will keep this one on my list of places to come back to when I am in AZ again.",
        "Never going back. This was my second time here. Wasn't that impressed the first time, but thought I'll give it another look, more because one of the guys in our team really wanted to go here. Food: Just ok, over-priced and very small portions. Came out hungry. Deserts are quite nice. Service: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed. Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed.",
        "Food: Just ok, over-priced and very small portions. Deserts are quite nice. Service: I think I would have given this place 2 12 stars was it not for the bad service. Very rude waiter, I am not going to go into details of what happenned at our table but suffice to say I left the place pissed. Never going back and nor is the guy in our team who really wanted to go in the first place.I think I hate a rude (and I mean rude not bad) service more that bad food. Bad food just leaves me with no feelings but a rude server leaves me pissed.",
    ]

    warmup = lm.preprocess(["The location is in a strip mall, but this place is a diamond in the rough."])
    model.predict(warmup, verbose=0)

    total_time = 0
    total_sentences = 0

    for review in reviews:
        conf = []
        times = []
        print(f'Review: {review}')
        review = review.split('.')
        context_buffer = []
        for sentence in review:
            if sentence.strip():
                context_buffer = ['.'.join(context_buffer + [sentence])]
                start_time = time.perf_counter()
                preprocessed_sentence = lm.preprocess(context_buffer)
                confidence = model.predict(preprocessed_sentence, verbose=0)[0][0]
                conf.append(confidence)
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)
                total_time += elapsed_time
                total_sentences += 1
                print(f"{elapsed_time:.4f} seconds")
                print(f"{confidence * 100:.2f}% positive")

        plt.figure(figsize=(12, 5))

        # График уверенности
        plt.subplot(1, 2, 1)
        plt.plot(conf, marker='o', color='b')
        plt.title('Confidence Levels')
        plt.xlabel('Sentence Index')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid()

        # График времени
        plt.subplot(1, 2, 2)
        plt.plot(times, marker='o', color='r')
        plt.title('Processing Times')
        plt.xlabel('Sentence Index')
        plt.ylabel('Time (seconds)')
        plt.ylim(0, 0.1)
        plt.grid()

        plt.tight_layout()
        plt.show()

        print('-----------------------------------------------------------------------------')

    print(f'Average processing time per sentence: {total_time/total_sentences:.4f} seconds')


if __name__ == '__main__':
    main()
