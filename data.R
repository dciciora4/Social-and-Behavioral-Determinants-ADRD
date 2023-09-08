# load important libraries

rm(list = ls())

options(scipen=999)
library(tidyverse)
library(stringr)
library(readxl)
library(janitor)
library(stringr)
library(purrr)
library(caret)
library(ggcorrplot)

#set wd

setwd("~/Desktop/masters_project/data_used")

# SVI 2018 data

svi_2018 <- read_csv("SVI2018_US_COUNTY.csv", 
                     col_types = cols(FIPS = col_double()))


svi_2018_vars <- svi_2018 %>% 
  select(ST, STATE, ST_ABBR, COUNTY, FIPS, E_TOTPOP, starts_with("RPL_THEME")) %>%  
  na_if(., -999) %>%  
  mutate(FIPS = sprintf("%05d", FIPS)) 



# CMS Prevalence data in percentages of beneficiaries  >= 65 yrs

CMS_over65_chronic <-
  read_excel(
    "County_Table_Chronic_Conditions_Prevalence_by_Age_2018.xlsx",
    sheet = "Beneficiaries 65 Years and Over",
    col_types = c(
      "text",
      "text",
      "text",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric",
      "numeric"
    ),
    skip = 5,
    na = "*"
  )

CMS_over65_chronic_vars <- CMS_over65_chronic %>% 
  filter(!is.na(`State/County FIPS Code`), # filter out rows for state summaries
         County != "Unknown") %>% # filter out unknown counties with no data
  select(`State/County FIPS Code`, `Alzheimer's Disease/Dementia`)



CMS_and_svi <- full_join(CMS_over65_chronic_vars, svi_2018_vars, by = c(`State/County FIPS Code` = "FIPS")) # join the two datasets




CMS_and_svi_2 <- CMS_and_svi %>% 
  rename(alzheimers_dementia = `Alzheimer's Disease/Dementia`,
         stcofips = `State/County FIPS Code`) %>% 
  relocate(alzheimers_dementia, .after = COUNTY)



# CMS Spending 2018

CMS_spending_2018 <- read_excel("County_Table_Chronic_Conditions_Spending_2018.xlsx", 
                                sheet = "Actual Spending", skip = 4, na = "*")
CMS_spending_2018_vars <- CMS_spending_2018 %>% 
  filter(!is.na(Fips),
         County != "Unknown") %>% 
  select(`Alzheimer's Disease/Dementia`, Fips) %>% 
  rename(alz_spending = `Alzheimer's Disease/Dementia`,
         stcofips = Fips )

# join CMS and svi

CMS_and_svi_all <- full_join(CMS_and_svi_2, CMS_spending_2018_vars, by = "stcofips") %>% 
  relocate(alz_spending, .after = alzheimers_dementia) %>% 
  clean_names()

## Read in CDC Data from cardiovascular source

zhang_data <- read_csv("CDC data 20220110_v2.csv")



zhang_data_vars <- zhang_data %>% 
  mutate(stcofips = sprintf("%05d", ffips),
         metro = as.factor(metro),
         log_homevalue = log(homevalue),
         white = replace(white, white == 262, 26.2)) %>%
  select(cholesterol, diabetes, smoking, obesity, inactive, nointernet,
         nocomputer, lesshighschool, lesscollege, snap, log_homevalue, hincome, gini, poverty, housingcost,
         unemploy, asian, black, white, pop, air, park, metro, hospital,
         physician, noinsurance, hisp, male, marri, crimerate, racialseg, socialvul, stcofips)

cms_svi_zhang <- full_join(CMS_and_svi_all, zhang_data_vars, by = "stcofips") %>% 
  select(-e_totpop, -rpl_theme1, -rpl_theme2, -rpl_theme3, -rpl_theme4, -rpl_themes)

#USGS Pesticide Usage... only keeping two with least amount of missing

pest <- read.delim("EPest_county_estimates_2019.txt", header = TRUE, sep = "\t", dec = ".")

pest_clean <- pest %>% 
  mutate(st_fips = sprintf("%02d", STATE_FIPS_CODE),
         co_fips = sprintf("%03d", COUNTY_FIPS_CODE),
         stcofips = str_c(st_fips, co_fips)) %>% 
  select(COMPOUND, EPEST_LOW_KG, EPEST_HIGH_KG, stcofips ) %>% 
  rowwise() %>% 
  mutate(est_pest = mean(c(EPEST_LOW_KG, EPEST_HIGH_KG), na.rm = T)) %>% 
  select(stcofips, COMPOUND, est_pest) %>% 
  pivot_wider(names_from = COMPOUND, values_from = est_pest) %>% 
  clean_names() %>% 
  select(stcofips, glyphosate, dicamba)

cms_svi_zhang_pest <- full_join(cms_svi_zhang, pest_clean, by = "stcofips")

hbp <- read_csv("hbp.csv")

hbp_vars <- hbp %>% 
  select(cnty_fips, Value) %>% 
  rename(stcofips = cnty_fips,
         hbp = Value)


## FEMA

fema <- read_csv("NRI_Table_Counties.csv")

fema_vars <- fema %>% 
  select(STCOFIPS, RISK_SCORE, EAL_SCORE, RESL_SCORE) %>% 
  clean_names()

cms_svi_zhang_pest_fema <- full_join(cms_svi_zhang_pest, fema_vars, by = "stcofips")


# EPA Walkability Index

epa_walk <- read_csv("EPA Walkability Index.csv")

epa_walk_vars <- epa_walk %>% 
  mutate(fips = sprintf("%011.0f", tract_id)) %>% 
  select(fips, natwalkind )

epa_walk_vars$stcofips <- str_sub(epa_walk_vars$fips, start = 1, end = 5)

epa_walk_agg <- epa_walk_vars %>% 
  group_by(stcofips) %>% 
  summarize(walk_index = mean(natwalkind, na.rm = T))

# CDC Food environment

cdc_food <- read_excel("CDC modified retail food index.xls")

cdc_food$stcofips <- str_sub(cdc_food$fips, start = 1, end = 5)

cdc_food_vars <- cdc_food %>% 
  select(stcofips, mrfei) %>% 
  group_by(stcofips) %>% 
  summarize(food_index = mean(mrfei, na.rm = T))


### Policy Map Data

policymap_heavydrinking <- read_csv("policymap_heavydrinking_2018.csv", na = "N/A")

policymap_heavydrinking_vars <- policymap_heavydrinking %>% 
  select(`Formatted FIPS`, `Estimated percent of adults reporting to engage in heavy drinkin`) %>% 
  rename(stcofips = `Formatted FIPS`, 
         hdrink = `Estimated percent of adults reporting to engage in heavy drinkin`)


fruit_veg_2017 <- read_csv("fruit_pm_2017.csv", na = "N/A")
fruit_veg_2017_vars <- fruit_veg_2017 %>% 
  select(`Formatted FIPS`, `Estimated percent of adults reporting to eat less than one servi`) %>% 
  rename(fruit = `Estimated percent of adults reporting to eat less than one servi`, 
         stcofips = `Formatted FIPS`) 

veteran_status <- read_csv("veteran_pm_20152019.csv", na = "N/A")
veteran_status_vars <- veteran_status %>% 
  select(`Formatted FIPS`, `Percent of civilians age 18 and older who are veterans, between`) %>% 
  rename(stcofips = `Formatted FIPS`, 
         veteran = `Percent of civilians age 18 and older who are veterans, between`)


vacant_house <- read_csv("vacant_pm_20152019.csv", na = "N/A")
vacant_house_vars <- vacant_house %>% 
  select(`Formatted FIPS`, `Estimated percent of housing units that were vacant in 2015-2019`) %>% 
  rename(stcofips = `Formatted FIPS`,
         vacant = `Estimated percent of housing units that were vacant in 2015-2019`)

incarcerate_2020 <- read_csv("incarcerate_pm_2020.csv", na = "N/A")
incarcerate_2020_vars <- incarcerate_2020 %>% 
  select(`Formatted FIPS`, `Percent of population living in correctional facilities for adul`) %>% 
  rename(stcofips = `Formatted FIPS`,
         incarcerate = `Percent of population living in correctional facilities for adul`)

manufacturing_2015_2019 <- read_csv("manufacturing_2015_2019.csv", na = "N/A") 

manufacturing_2015_2019_vars <- manufacturing_2015_2019 %>% 
  select(`Formatted FIPS`, `Estimated percent of people age 16 years or older who were emplo`) %>% 
  rename(stcofips = `Formatted FIPS`,
         manufacturing = `Estimated percent of people age 16 years or older who were emplo`)

widow_2015_2019 <- read_csv("widow_2015_2019.csv", na = "N/A")

widow_2015_2019_vars <- widow_2015_2019 %>% 
  select(`Formatted FIPS`, `Estimated percent of people age 15 and over who are widowed and`) %>% 
  rename(stcofips = `Formatted FIPS`,
         widow = `Estimated percent of people age 15 and over who are widowed and`)


hearing_2015_2019 <- read_csv("hearing_2015_2019.csv", na = "N/A")

hearing_2015_2019_vars <- hearing_2015_2019 %>% 
  select(`Formatted FIPS`, `Estimated percent of people with a hearing difficulty, between 2`) %>% 
  rename(stcofips = `Formatted FIPS`,
         hearing = `Estimated percent of people with a hearing difficulty, between 2`)


# EJSCREEN Data

EJSCREEN_2020_USPR <- read_csv("EJSCREEN_2020_USPR.csv")


airdata_2020 <- EJSCREEN_2020_USPR %>%
  select(c("ID", "ACSTOTHU", "PRE1960", "PRE1960PCT", "DSLPM", "RESP", "PTRAF", "PTSDF", "OZONE", "PM25" )) # initial columns of interest

# get county fips from block
airdata_2020$ID <- str_sub(airdata_2020$ID, start = 1, end = 5)


airdata_2020_county <- airdata_2020 %>% 
  group_by(ID) %>% # averages for each block group to get county # prob need to weight them
  summarize(tot_hu = sum(ACSTOTHU, na.rm = T),
            tot_1960 = sum(PRE1960, na.rm = T),
            pre1960pct = tot_1960/tot_hu,
            across(c(DSLPM:OZONE), ~ mean(.x, na.rm = TRUE))) %>% 
  janitor::clean_names() %>% 
  rename(stcofips = id) %>% 
  select(stcofips, pre1960pct, dslpm, resp, ptraf, ptsdf, ozone)



### County Health Rankings data

county_health_rankings <- read_csv("analytic_data2021 (1).csv", 
                                   skip = 1)
county_health_rankings_vars <- county_health_rankings %>% 
  select(fipscode, v132_rawvalue, v140_rawvalue, v143_rawvalue, v145_rawvalue) %>% 
  rename(access_exercise = v132_rawvalue,
         social_assoc = v140_rawvalue,
         sleep = v143_rawvalue,
         mental = v145_rawvalue,
         stcofips = fipscode)



#CDC Atlas Plus 

chlamydia <-  read_csv("AtlasPlusTableData.csv", na = "Data not available") %>% 
  rename(stcofips = FIPS,
         chlamydia = `Chlamydia Rate per 100000`)



syphilis <- read_csv("syphillis_2018.csv", na = "Data not available") %>% 
  rename(syphilis = primary_secondary_syphillis_rate)


### CDC PLACES

cdc_places <- read_csv("PLACES__Local_Data_for_Better_Health__County_Data_2021_release.csv")

cdc_places_vars <- cdc_places %>% 
  filter(Year == 2018) %>% 
  select(Measure, Data_Value, Data_Value_Type, LocationID) %>% 
  filter(Measure == "All teeth lost among adults aged >=65 years",
         LocationID != "59",
         Data_Value_Type == "Age-adjusted prevalence") %>% 
  select(-Data_Value_Type, -Measure) %>% 
  rename(teeth = Data_Value,
         stcofips = LocationID)

## All data

all_data <- list(cms_svi_zhang_pest_fema, epa_walk_agg, cdc_food_vars, policymap_heavydrinking_vars,
                 fruit_veg_2017_vars, veteran_status_vars, vacant_house_vars, incarcerate_2020_vars,
                 airdata_2020_county, county_health_rankings_vars, chlamydia, syphilis, manufacturing_2015_2019_vars,
                 hbp_vars, widow_2015_2019_vars, hearing_2015_2019_vars, cdc_places_vars) %>% 
  reduce(full_join, by = "stcofips")



all_data <- all_data %>% 
  filter(stcofips != "00000",
         stcofips != "000NA",
         stcofips != "01000") %>% 
  mutate(st = case_when(
    alzheimers_dementia == 0 ~ str_sub(stcofips, start = 1, end = 2), 
    TRUE   ~ st 
  )) %>% 
  filter(!is.na(st))



### Remove variables


write.csv(all_data, "~/Desktop/masters_project/data_used/all_data_12apr2022.csv", row.names = F)
