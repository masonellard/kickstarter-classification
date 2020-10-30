\connect kickstarter;

\copy ( SELECT * FROM (SELECT *, CAST(SUBSTR(deadline, 1, 4) AS INT) AS year FROM projects) AS Table1 WHERE year >= 2014 ORDER BY RANDOM() LIMIT 20000) To '~/Metis/project-3/kickstarter_data_update.csv' With CSV DELIMITER ',' HEADER
