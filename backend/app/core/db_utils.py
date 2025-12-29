
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal

async def upsert_object(session: AsyncSession, model, values: dict, index_elements: list):
    """
    Upsert (Insert or Update) an object into the database.
    
    Args:
        session: SQLAlchemy AsyncSession
        model: SQLAlchemy Model class
        values: Dictionary of column values
        index_elements: List of column names that form the unique constraint (e.g. ['time', 'symbol'])
    """
    stmt = insert(model).values(values)
    
    # Create update dictionary (update all columns except primary keys)
    update_dict = {
        col.name: col 
        for col in stmt.excluded 
        if col.name not in index_elements
    }
    
    if update_dict:
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=update_dict
        )
    else:
        # If there are no columns to update (e.g. only PKs), do nothing
        stmt = stmt.on_conflict_do_nothing(
            index_elements=index_elements
        )
        
    await session.execute(stmt)
