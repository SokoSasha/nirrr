import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

from lstm_model import BestModelEverLOL
from text_processor import LanguageModel


def main():
    model = BestModelEverLOL.load("lstm_model_stateful.keras")
    lm = LanguageModel.load()
    msl = model.get_max_sequence_length

    # Список отзывов для обработки
    reviews = [
        "A NOTE TO THE OWNER:Your employee, the scrawny, balding man, should be fired. He was yelling at me and saying extremely rude things instead of fixing the nails to make the customer happy. I've never been treated this way by any company that I've ever done business with. I urge you to take a few moments to read about what happened from the customer's point of view, as I'm sure that you may hear about it but from a different light that is favorable to the scrawny, balding man.MY REVIEW: They shouldn't even get ONE STAR. The scrawny man with the receding hair that works there is the ultimate cause of this poor review.Sit back and enjoy this story...This was my very first time at this place and I was actually excited because the interior was pretty nice.I went in to touch up my gel polish and get a pedicure. I get French on both, and I've gotten it done enough times to know that the white part should not have BUBBLES or CRATERS in it! So I kindly asked him to fix the white part before we put it under the UV light again. So he did, and we put it under the light, when I took them out...there they were...craters, bubbles, missing white parts, streaks...(how do you post pics on here) I swear it looked like I did it myself. He said it's supposed to look like that.Instead of fixing them, he yelled at me...I mean...he YELLED AT ME worse than my own father ever has. He was so rude, and the things he said to me were jaw dropping! I honestly was in shock and could not believe my ears! Now, I know that people say that there are two sides to every story, but I promise you that I wasn't happy with the white part and asked him to fix it and he flipped out and started taking it off! As a customer...I've never ever been talked to, oh I'm sorry, YELLED AT, like that in my entire life. I told him, ""I've never been treated this way in my life. You're being very rude to a first time customer!"" He said, ""Ive never had a customer tell me how to do my job!""I said, I'm just asking you to fix the white, LOOK can't you see the dots Does this honestly look okay to you"" Then he started to file it all offf.I was like ""what are you doing, just fix the white!"" He said, ""No I'm gonna take it off and you can just leave""I said, ""What No, I came here to get my nails done and I just don't want the french white part to have bubbles or dots on it.""He yelled, ""I've been doing this 20 years, how old are you (I don't know what my age has to do with it, but I suppose he was trying to say that he's been doing this longer than I've been living."" (for the record I'm 28 and not an immature teenager bashing this place) Anyway, so the yelling at me continued and at this point I begin to yell back because I was not going to take the verbal abuse or mistreatment any longer. I've never experienced such an episode.He had the UV thing on my lap, I tried to hand it to him and I said please get this off of me, and he slid his seat away and was telling me to drop it so it can break! Seriously! drop itThen I said, ""I want to talk to the owner of this place. You are so rude and should not be treating customers this way!""He YELLED at the top of his lungs like he wanted to hit me...""NO! YOU ARE NOT GOING TO TALK TO THE OWNER, YOU DEAL WITH ME I'M THE BOSS NOW!!"" I was like WHAT IN THE WORLD IS YOUR PROBLEM I just wanted you to fix the white part of my nails!Then he said, ""pay for the pedicure and get out of here and we will be happy if you never come back!"" (can you believe this guy!)I don't know how this next part even happened (and this literally JUST HAPPENED) but he said he was going to call the cops, I was like good, please do, or better yet, I will. So the cops came. (yes, seriously)I told them and SHOWED them my hands of how poor of a job he did and the way he treated me. The cops obviously can't do anything about this situation so it was dumb even having them come. The guy didn't even have anything to say but, ""She is trying to tell me how to do my job"" It's like whatever happened to trying to get the nails like the customer likes them Especially a new customerAnyway, so I paid them for the pedicure and left with the nails you're about to see...I DO NOT RECOMMEND THIS PLACE AT ALL. NOT EVEN IF THEY OFFER A FREE MANICURE AND PEDICURE. THEY DO A SLOPPY JOB.THEY ARE PURELY MEAN AND RUDE TO THEIR CUSTOMERS.The guy seriously yelled at me for at least 10 minutes, maybe more. I have never been so disrespected in my life.I was literally crying because of this man! If you were there, your ears would have perked up and you would have been so shocked to hear the things he was saying and THE WAY he was saying them.I don't think I'm even painting the picture of how evil he is as best as I could. I wish I could recall all of the insanely rude things that he lashed out at me with.Oh, and they don't take credit cards by the way...they have a lovely ATM machine for your convenience (please take note of my sarcasm)",
        "Worst place I've ever been to    No service or women. I would never come in this place ever again.  Women are not attractive and staff is even worse.Beer is almost $8 with no girls in sight.  What a joke.  This place is the worst place I've ever been to and I've been all over the world.",
        "Dianne is awesome, she has painted some awesome pictures for me at a few events.  Highly recommended.",
        "I can't stress enough how great this place is.  I've been coming here regularly since I discovered it last winter and will continue to do so. The first thing I noticed about this place was their great customer service.  The workers are always friendly - especially the owners - and you can tell that they try to get to know a lot of their regular customers.  They're also very helpful in answering any questions about the teas and can give recommendations.  Thumbs up on the milk tea, mountain tea (a really unique and great taste), coffee and the fruit teas.  The little desserts are a nice touch and the one I've tried was good.  Their steamed buns are good as well.The only thing I wouldn't recommend is the pork ramen.  My friend and I both tried it and neither of us liked how fatty the pork was.  The ramen is okay but I probably wouldn't order it again for the price.  But overall I find their prices to be very reasonable.",
    ]

    warmup = pad_sequences(
        lm.texts_to_sequences(["The location is in a strip mall, but this place is a diamond in the rough."]),
        maxlen=msl)
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
                start_time = time.perf_counter()
                context_buffer.extend(lm.texts_to_sequences([sentence])[0])
                lcb = len(context_buffer)
                if lcb > msl:
                    context_buffer = context_buffer[lcb - msl:]
                sequences_padded = lm.pad_sequences([context_buffer], maxlen=msl)
                confidence = model.predict(sequences_padded, verbose=0)[0][0]
                conf.append(confidence)
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)
                total_time += elapsed_time
                total_sentences += 1
                # print(f"{elapsed_time:.4f} seconds")
                # print(f"{confidence * 100:.2f}% positive")
        model.reset()

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

    print(f'Average processing time per sentence: {total_time / total_sentences:.4f} seconds')


if __name__ == '__main__':
    main()

# классификация моделей
# фишинг шире
# почему без датасета
# использование
