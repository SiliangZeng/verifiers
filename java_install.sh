# 首先删除旧版本的 Java
apt-get remove --purge openjdk*

# 添加 Java 21 的仓库
apt-get update
apt-get install -y wget gpg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list

# 安装 Java 21
apt-get update
apt-get install -y temurin-21-jdk

# 设置环境变量
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export JAVA_OPTS="--add-modules jdk.incubator.vector -Xms4g -Xmx12g -XX:+UseG1GC"

# 验证 Java 安装
java -version

# 设置必要的环境变量
export JVM_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server/libjvm.so
export LD_LIBRARY_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server:$LD_LIBRARY_PATH