import { useState, createContext, useMemo } from 'react';

const UserContext = createContext();

const UserProvider = (props) => {
  const [username, setUsername_] = useState(
    window.localStorage.getItem("foragerUsername") || ""
  );
  const setUsername = (u) => {
    window.localStorage.setItem("foragerUsername", u.trim());
    setUsername_(u);
  };

  const value = useMemo(
    () => ({username, setUsername}),[username])

  return (
    <UserContext.Provider
      value={value}
    >
      {props.children}
    </UserContext.Provider>
  );
}

export { UserContext, UserProvider };
