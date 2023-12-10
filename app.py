
from flask import Flask
# from flask_restful import reqparse
from flask_restful import Api, Resource

from modules.UserHandler import CreateUser, ChangePassword, Login, GetUsers
from modules.LrpHandle import ImageReg, VideoReg, GetVidInfo
from modules.ViPhamHandle import CheckViPham, ThongKe

app = Flask(__name__)
api = Api(app)

api.add_resource(CreateUser, "/user/create-user")
api.add_resource(ChangePassword, "/user/change-password")
api.add_resource(Login, "/user/login")
api.add_resource(ImageReg, "/reg/image")
api.add_resource(VideoReg, "/reg/video")
api.add_resource(GetVidInfo, "/log/info/")
api.add_resource(GetUsers, "/user/get-users/")
api.add_resource(ThongKe, "/vi-pham/thong-ke/")
api.add_resource(CheckViPham, "/vi-pham/check/")

if __name__ == '__main__':
    app.run("0.0.0.0", 5000)
