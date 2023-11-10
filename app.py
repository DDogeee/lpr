
from flask import Flask
# from flask_restful import reqparse
from flask_restful import Api, Resource

from modules.UserHandler import CreateUser, ChangePassword, Login
from modules.LrpHandle import ImageReg, VideoReg, GetVidInfo

app = Flask(__name__)
api = Api(app)

api.add_resource(CreateUser, "/user/create-user")
api.add_resource(ChangePassword, "/user/change-password")
api.add_resource(Login, "/user/login")
api.add_resource(ImageReg, "/reg/image")
api.add_resource(VideoReg, "/reg/video")
api.add_resource(GetVidInfo, "/log/info/")

if __name__ == '__main__':
    app.run("0.0.0.0", 5000)
