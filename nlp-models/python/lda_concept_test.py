from lda_concept import LDA


if __name__ == '__main__':
    documents = [line.rstrip() for line in open('temp/all_book_titles.txt')]

    stopwords = set(line.rstrip() for line in open('temp/stopwords.txt')).union({
        'introduction', 'edition', 'series', 'application',
        'approach', 'card', 'access', 'package', 'plus', 'etext',
        'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
        'third', 'second', 'fourth',
    })

    model = LDA(stopwords)
    model.fit(documents)
    model.concepts()
