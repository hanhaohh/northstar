from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String


Base = declarative_base()


class FileMetadata(Base):

    __tablename__ = "file_metadata"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(128))
    file_path = Column(String(128))
    file_md5 = Column(String(128))

    def __repr__(self):
        return "<metadata(name='%s', fullname='%s')>" % (
            self.file_name,
            self.file_md5
        )
 
