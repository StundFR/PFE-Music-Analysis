
CREATE TABLE artist(
   id_artist INT GENERATED ALWAYS AS IDENTITY,
   nameArtist VARCHAR(100),
   creationDate DATE,
   city VARCHAR(100),
   country VARCHAR(100),
   PRIMARY KEY(id_artist)
);

CREATE TABLE music(
   id_music INT GENERATED ALWAYS AS IDENTITY,
   id_artist INT,
   title VARCHAR(100),
   lyrics VARCHAR(1000),
   genre VARCHAR(100),
   releaseDate DATE,
   ranking INT,
   PRIMARY KEY(id_music),
   CONSTRAINT fk_artist
      FOREIGN KEY(id_artist) 
	  REFERENCES artist(id_artist)
	  ON DELETE CASCADE
);