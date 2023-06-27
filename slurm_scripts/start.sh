# Delete file nianetcae.sh if it exists
rm -f nianetcae.sh
# wget file from url into current folder
wget https://raw.githubusercontent.com/SasoPavlic/NiaNetCAE/main/nianetcae.sh
# Change permissions to 777
chmod 777 nianetcae.sh
# Delete folders logs,data, configs if they exist
rm -rf logs data configs
# Change permissions to 777
mkdir -m 777 logs data configs
# Check if data.zip exists otherwise download it from url
if [ ! -f data.zip ]; then
    wget https://github.com/SasoPavlic/NiaNetCAE/blob/main/data/nyu_data.zip
fi
# Unzip content of data.zip to folder data
unzip data.zip -d data
# wget file from url into config folder
wget https://github.com/SasoPavlic/NiaNetCAE/blob/main/configs/main_config.yaml -P configs
# Check if parameter is passed to script
if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Usage: ./start.sh <number_of_runs>"
    echo "Example: ./start.sh 10"
    exit 1
else
  # Run the job N times based on parameter passed into this script
  for i in $(seq 1 $1); do
      # Run nianetcae.sh
      sbatch nianetcae.sh
  done
fi