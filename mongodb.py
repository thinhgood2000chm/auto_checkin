import pymongo

myclient = pymongo.MongoClient("mongodb://192.168.10.90:27017/")

db = myclient['ekyc']
print("connect mongodb ")

def get_all_embedding(corpcode):
    return db.embedding.find({"corp_code": corpcode, "is_activate": True})


def query_user_info(corp_code, user_code):
    cursor = db.user.find_one({"corp_code": corp_code, "user_code": user_code, "is_activate": True})
    return cursor