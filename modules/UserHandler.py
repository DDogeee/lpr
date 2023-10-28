from flask_restful import Resource, reqparse
from pymongo import MongoClient
from config import URI


class UserHandler(Resource):
    def __init__(self) -> None:
        self.collection = MongoClient(URI).main.users
        args = reqparse.RequestParser()
        args.add_argument("username", type=str, required=True, help="username is missing")
        args.add_argument("password", type=str, required=True, help="password is missing")
        self.args = args

    def create_user(self, username: str, password: str):
        user_ = self.collection.find_one({
            "username": username
        })
        if user_ is not None:
            return {
                "error": True,
                "message": "Username already exists",
                "data": None
            }

        elif len(password) == 0:
            return {
                "error": True,
                "message": "Blank password",
                "data": None
            }
        else:
            item = {
                "username": username,
                "password": password,
            }
            self.collection.insert_one(item)
            return {
                "error": False,
                "message": "User created",
                "data": None
            }

    def change_password(self, username: str, old_password: str, new_password: str):
        user_ = self.collection.find_one({
            "username": username
        })
        if user_ is None:
            return {
                "error": True,
                "message": "Username not exists",
                "data": None
            }
        elif user_['password'] != old_password:
            return {
                "error": True,
                "message": "Wrong password",
                "data": None
            }
        else:
            item = {
                "username": username,
            }

            new_item = {"$set": {"password": new_password}}

            self.collection.update_one(item, new_item)
            return {
                "error": False,
                "message": "Password changed",
                "data": None
            }
        
    def login(self, username: str, password: str):
        user_ = self.collection.find_one({
            "username": username,
            "password" : password
        })
        if user_ is not None:
            return {
                "error": False,
                "message": "Login succesful",
                "data": None
            }
        else:
            return {
                "error": False,
                "message": "Wrong password or user not found",
                "data": None
            }

class CreateUser(UserHandler):
    def post(self):
        args = self.args.parse_args()
        return self.create_user(args['username'], args['password'])
        
class ChangePassword(UserHandler):
    def post(self):
        args = self.args
        args.add_argument("new_password", type=str, required=True, help="new password is missing")
        args = args.parse_args()
        return self.change_password(args['username'], args['password'], args['new_password'])

class Login(UserHandler):
    def post(self):
        args = self.args.parse_args()
        return self.login(args['username'], args['password'])