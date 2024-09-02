""" Dict Object for Index Mapping in elasticsearch """
indexName = "document_embedding"

# indexMapping = {
#     "properties": {
#         "paper_id": {
#             "type": "text"
#         },
#         "doc_id": {
#             "type": "text"
#         },
#         "title": {
#             "type": "text"
#         },
#         "doc": {
#             "type": "text"
#         },
#         "DocEmbedding": {
#             "type": "dense_vector",
#             "dims": 384,
#             "index": True,
#             "similarity": "cosine"
#         }
#
#     }
# }


indexMapping = {
    "properties": {
        "doc_id": {
            "type": "text"
        },
        "DocEmbedding": {
            "type": "dense_vector",
            "dims": 384,
            "index": True,
            "similarity": "cosine"
        }

    }
}