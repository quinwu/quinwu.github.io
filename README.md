## 为什么会有这个项目

这个项目是自己用github pages 搭建的一个静态[blog](quinwu.org)。master分支为为blog的页面部署的相关js，html文件，hexo分支为本地的一些源文件，记录下本地的一些配置信息跟博文的*.md源文件，保证在其他的电脑上可以无差别的部署。

这个博客的部署要感谢

- [hexo](https://hexo.io/) 一个快速，简单 ，文档的静态博客框架。
- [next](http://theme-next.iissnan.com/) 简单，精巧的Hexo 主题



## 部署信息

```shell
#hexo Markdown file
git clone git@github.com:quinwu/quinwu.github.io.git  hexo/blog 
cd hexo/blog/themes/
#hexo-next themes file
git clone git@github.com:quinwu/hexo-theme-next.git nextclear
```

安装node.js

安装hexo

```shell
 npm install -g hexo-cli
 npm install hexo --save
 hexo -v
 hexo init
 npm install -- save-dev hexo-util #缺少hexo-util时
 npm install 
 
 
 #hexo 常用的生产部署命令
 hexo g 
 hexo s
 hexo d
 hexo clean
```

---

# 快速脚本

`local.sh`本地调试

`deploy.sh`部署到github/gitcafe

---

# 相关配置的备份

### `backup_config`目录下

#### clone theme next

clone theme-next address:

```shell
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

#### https open
```shell
cp backup_config/_layout.swig      /themes/next/layout/_layout.swig
```

#### my_avatar.jpg

`my_avatat.jpg`    头像

```shell
cp my_avatar.jpg    ../themes/next/source/images/
```

#### alipay.jpg  wechatpay.jpg

`alipay.jpg` 支付宝二维码

`wechatpay.jpg` 微信二维码

```shell
cp *pay.jpg ../themes/next/source/images
```

#### blog_config.yml 

`blog_config.yml `站点配置文件 

```shell
cp blog_config.yml ../_config.yml
```

#### next_config.yml

`next_config.yml` 主题配置文件

```shell
cp next_config.yml ../themes/next/_config.yml
```

#### marked.js

`marked.js` 修改过的js备份文件（markdown 与 latex的 `_` `\\`语法冲动，修改后markdown 斜体 `_text_` `*test*`无法解析）不推荐这种方式

```shell
cp marked.js ../node_modules/marked/lib
```

#### backup_marked.js

`backup_marked.js` 原未被修改的js备份文件

```shell
cp backup_marked.js ../node_modules/marked/lib/marked.js
```







