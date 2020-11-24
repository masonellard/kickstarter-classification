\connect kickstarter;

CREATE TABLE Projects(
	ID INTEGER,
	name TEXT,
	category TEXT,
	main_category TEXT,
	currency TEXT,
	deadline TEXT,
	goal FLOAT,
	launched TEXT,
	pledged FLOAT,
	state TEXT,
	backers INTEGER,
	country TEXT,
	usd_pledged FLOAT,
	usd_pledged_real FLOAT,
	usd_goal_real FLOAT
);

\copy Projects FROM 'ks-projects-201801.csv' DELIMITER ',' CSV HEADER;

