# InformationRetrievalSystem
Fall 2021

A dictionary is made by the documents and its words are used as inverted indexes. Then documents are mapped into indexes' space by using tf-idf term before being clustered into certain groups. Queries are mapped into the same space and responded through seeking among documents in the relevant cluster by calculating the cosine similarity and inserting them into a max-heap. Finally top K relevant documents are retrieved from the heap and displayed in the console.
