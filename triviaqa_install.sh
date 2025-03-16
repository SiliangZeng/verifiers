# Install Java 21 (required for Pyserini-Search), if you want to run the TriviaQA search example
# First remove any old Java versions
apt-get remove --purge openjdk*

# Add Java 21 repository
apt-get update
apt-get install -y wget gpg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list

# Install Java 21
apt-get update
apt-get install -y temurin-21-jdk

# Set Java environment variables
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export JAVA_OPTS="--add-modules jdk.incubator.vector -Xms4g -Xmx12g -XX:+UseG1GC"
export JVM_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server/libjvm.so
export LD_LIBRARY_PATH=/usr/lib/jvm/temurin-21-jdk-amd64/lib/server:$LD_LIBRARY_PATH

# Verify Java installation
java -version

# Create a virtual environment named "triviaqa-env"

uv sync

uv pip install pyserini

# Install flash-attn separately (requires special handling)
uv pip install flash-attn --no-build-isolation