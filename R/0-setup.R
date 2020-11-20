# install the latest version from CRAN
install.packages("azuremlsdk")
azuremlsdk::install_azureml(envname = 'r-reticulate')

# Install additional packages that will be used in this module
install.packages(c('data.table'))

library(azuremlsdk)