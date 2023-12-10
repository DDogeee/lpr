from flask_restful import Resource, reqparse
from pymongo import MongoClient
from config import URI, PRIVATE_KEY
import jwt


class ViPhamHandler(Resource):
    def check_vi_pham(self, plate: str, token: str):
        collection = MongoClient(URI).main.users
        try:
            user = jwt.decode(token, PRIVATE_KEY, algorithms=["HS256"])
        except:
            return {
                "error": True,
                "message": "Invalid token",
                "data": None
            }
        user_ = collection.find_one({
            "username": user['username'],
            "password" : user['password']
        })
        if user_ is not None:
            if user_['role'] != 'admin':
                return {
                    "error": True,
                    "message": "Not enough permission",
                    "data": None
                }
        plate_ = self.collection.find({
            "plate": plate
        }, {"_id" : 0})
        plate_ = [plate for plate in plate_]
        if plate_ is not None:
            return {
                "speeding": True,
                "data": plate_[:20]
            }
        else:
            return {
                "speeding": False,
                "data": None
            }
        
    def thong_ke_vi_pham(self, date: str, token: str):
        collection = MongoClient(URI).main.users
        try:
            user = jwt.decode(token, PRIVATE_KEY, algorithms=["HS256"])
        except:
            return {
                "error": True,
                "message": "Invalid token",
                "data": None
            }
        user_ = collection.find_one({
            "username": user['username'],
            "password" : user['password']
        })
        if user_ is not None:
            if user_['role'] != 'admin':
                return {
                    "error": True,
                    "message": "Not enough permission",
                    "data": None
                }
        date_ = self.collection.find({
            "date": date
        }, {"_id" : 0})
        date_ = [d for d in date_]
        if date_ is not None:
            return {
                "data": date_[:20]
            }
        else:
            return {
                "data": None
            }
class ThongKe(ViPhamHandler):
    def __init__(self):
        self.collection = MongoClient(URI).main.vi_pham
        args = reqparse.RequestParser()
        args.add_argument("date", type=str, required=True, help="date is missing")
        args.add_argument("token", type=str, required=True, help="token is missing")
        self.args = args
    def post(self):
        args = self.args
        args = args.parse_args()
        return self.thong_ke_vi_pham(args['date'], args['token'])

class CheckViPham(ViPhamHandler):
    def __init__(self):
        self.collection = MongoClient(URI).main.vi_pham
        args = reqparse.RequestParser()
        args.add_argument("plate", type=str, required=True, help="plate is missing")
        args.add_argument("token", type=str, required=True, help="token is missing")
        self.args = args
    def post(self):
        args = self.args
        args = args.parse_args()
        return self.check_vi_pham(args['plate'], args['token'])
